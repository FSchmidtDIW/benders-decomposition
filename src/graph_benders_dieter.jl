using DIETER
using JuMP
#using Plasmo
using Gurobi
#using DSPopt
#using StructJuMP
#using MPI
using Distributed



mutable struct StabObject
    method::Union{Symbol,Nothing}
    method_parameters::Dict{Symbol,Any}
    dynPar::Array{Union{Dict,Float64},1}
    function StabObject(method::Union{Symbol,Nothing})
        stab_obj = new()
        stab_obj.method = method
        stab_obj.method_parameters = Dict{Symbol,Any}()
        stab_obj.dynPar = Union{Dict,Float64}[]
        return stab_obj
    end
end

mutable struct resData
	objVal::Float64
	capa::Dict{Symbol,Dict{Symbol,Dict{Symbol,DataFrame}}}
	stLvl::Dict{Symbol,DataFrame}
	resData() = new(Inf,Dict{Symbol,Dict{Symbol,Dict{Symbol,DataFrame}}}(),Dict{Symbol,DataFrame}())
end




mutable struct MasterProblem
    model::JuMP.Model
    dtr::DieterModel
    stab_obj::StabObject
    vars::Dict{Symbol,Union{Vector{VariableRef},Nothing}}
    first_stage_obj::AffExpr
    stab_centre::Dict{Symbol,Union{Vector{Float64},Nothing}}
    cuts::AbstractDataFrame
    function MasterProblem(optimizer,datapath::String;stab_method::Union{Symbol,Nothing}=nothing)
        master_problem = new()
        master_problem.model = Model(optimizer)
        master_problem.dtr = DieterModel(datapath,verbose=false);
        master_problem.stab_obj = StabObject(stab_method)
        master_problem.vars = Dict(:capa => nothing,:Stlvl => nothing,:h2_import => nothing)
        master_problem.first_stage_obj = AffExpr()
        master_problem.stab_centre = Dict(:capa => nothing,:Stlvl => nothing,:h2_import => nothing)
        master_problem.cuts = DataFrame(iteration = Int64[],year=Union{Int64,String}[],cut=ConstraintRef[],last_active=Union{Int64,Missing}[])
        return master_problem
    end
end



function compile_master_problem!(master::MasterProblem,settings::NamedTuple)
    # Define model
    define_model!(master.model,master.dtr,maxhours=0)
    set_attribute(master.model,"OutputFlag",0);
    
    # First stage objective c^T x
    master.first_stage_obj = objective_function(master.model)

    # Sets of complicating variables
    all_vars = all_variables(master.model)
    master.vars[:h2_import] = settings.h2_import_master ? filter(x->occursin("H2_IMP",string(x)),all_vars) : nothing
    master.vars[:capa] = setdiff(all_vars,master.vars[:h2_import])
    # Add variables for storage levels
    if settings.lds_mode
        @variables(master.model,begin
            Stlvl[n=master.dtr.sets[:n],s=master.dtr.sets[:stoh2],y=settings.year_set; DIETER.cond_ext(:h2,n,master.dtr) && DIETER.cond_h2(s,n,master.dtr) && master.dtr.parameters[:h2_max_energy_sto][n,s] != 0] >= 0
        end);
        @constraint(master.model,[n=master.dtr.sets[:n],s=master.dtr.sets[:stoh2],y=settings.year_set; DIETER.cond_ext(:h2,n,master.dtr) && DIETER.cond_h2(s,n,master.dtr) && master.dtr.parameters[:h2_max_energy_sto][n,s] != 0],Stlvl[n,s,y] <= master.model[:H2_N_STO_E][n,s])
        master.vars[:Stlvl] = setdiff(all_variables(master.model),all_vars)
    end

    # Add cutting plane model
    probs = DenseAxisArray(settings.probabilities,settings.year_set)
    if settings.multicut_bool
        @variable(master.model, θ[settings.year_set] >= 0);
        @expression(master.model, estimator, sum(probs[y]*θ[y] for y in settings.year_set));
        @objective(master.model, Min, master.first_stage_obj + estimator);
        
    else
        @variable(master.model, θ >= 0);
        @objective(master.model, Min, master.first_stage_obj + θ);
    end
end



function collect_iterate(master::MasterProblem)
    res = Dict(k => DataFrame(:symbol => Symbol.(master.vars[k]),
                             :master_var => master.vars[k],
                             :value => value.(master.vars[k])) 
                             for k in keys(master.vars))
    return res
end


function update_stab_centre!(master::MasterProblem)
    for var in keys(master.vars)
        if !isnothing(master.vars[var]) 
            master.stab_centre[var] = DataFrame(:var_name => Symbol.(master.vars[var]), :centre => value.(master.vars[var]))
        end
    end
end



mutable struct SubProblem
    year::Union{Int64,String}
    model::JuMP.Model
    dtr::DieterModel
    vars::Dict{Symbol,Union{DataFrame,Nothing}}
    function SubProblem(year,optimizer,datapath::String)
        sub_problem = new()
        sub_problem.year = year
        sub_problem.model = Model(optimizer)
        sub_problem.dtr = DieterModel(datapath,verbose=false);
        sub_problem.vars = Dict(:capa => nothing,:Stlvl => nothing,:h2_import => nothing)
        return sub_problem
    end
end


function map_storage_vars!(sub::SubProblem,settings)
    year = sub.year
    pre_year = DIETER.preceding_period(settings.year_set,year)
    tmp = DIETER.convert_jump_container_to_df(sub.model[:H2_STO_L],dim_names = [:n,:s,:h],value_name = :var_name)
    tmp_start = filter(row->(row.h == "t0001"),tmp)
    tmp_end = filter(row->(row.h == "t"*lpad(settings.maxhours,4,"0")),tmp)
    tmp_start[!,:symbol] = Symbol.(string.("Stlvl[",tmp_start.n,",",tmp_start.s,",",pre_year,"]"))
    tmp_end[!,:symbol] = Symbol.(string.("Stlvl[",tmp_start.n,",",tmp_start.s,",",year,"]"))
    select!(tmp_start,[:var_name,:symbol])
    select!(tmp_end,[:var_name,:symbol])
    sub.vars[:Stlvl] = [tmp_start;tmp_end]
end



function build_sub_problem!(sub::SubProblem,settings::NamedTuple,vars::Dict)
    # create DieterModel with correct data
    sub.dtr = create_sub_dict(sub.dtr,sub.year,settings.ts_path)
    # define JuMP Model 
    define_model!(sub.model,sub.dtr,maxhours=settings.maxhours)
    # Define complicating variables
    sub.vars[:capa] = DataFrame(:var_name => filter(x-> string(x) in string.(vars[:capa]),all_variables(sub.model)))    
    sub.vars[:capa][!,:symbol] = Symbol.(sub.vars[:capa][!,:var_name])
    if settings.h2_import_master
        sub.vars[:h2_import] = settings.h2_import_master ? DataFrame(:var_name => filter(x-> string(x) in string.(vars[:h2_import]),all_variables(sub.model))) : nothing
        sub.vars[:h2_import][!,:symbol] = Symbol.(sub.vars[:h2_import][!,:var_name])
    end
    if settings.lds_mode
        map_storage_vars!(sub,settings);
    end
end

mutable struct Oracle
    objVal::Float64
    duals::Dict{Symbol,DataFrame}
    function Oracle(solved_mod::JuMP.Model,sub::SubProblem)
        oracle = new()
        oracle.objVal = objective_value(solved_mod)
        oracle.duals = Dict{Symbol,DataFrame}()
        for k in keys(sub.vars)
            if !isnothing(sub.vars[k])
                oracle.duals[k] = DataFrame(:symbol => Symbol.(sub.vars[k][!,:symbol]),
                                            #:master_var => Symbol.(sub.vars[k][!,:symbol]),
                                            :dual => dual.(sub.vars[k][!,:cons]))
            end
        end
        return oracle
    end
end



function fix_complicating_vars!(model::JuMP.Model,sub::SubProblem,res::Dict,type::Symbol)
    df = leftjoin(sub.vars[type], res[type], on = :symbol)
    model[Symbol("fix_$(type)")] = @constraint(model,df[!,:var_name] .== df[!,:value])
    sub.vars[type][!,:cons] = model[Symbol("fix_$(type)")]
end

function solve_subproblem(sub::SubProblem,res::Dict,opt_gap::Float64,settings::NamedTuple)
    # Create temporary model instance
    temp_sub = deepcopy(sub)
    temp_mod = temp_sub.model
    set_optimizer(temp_mod,Gurobi.Optimizer)
    set_silent(temp_mod)
    set_optimizer_attribute(temp_mod,"OutputFlag",0)
    set_optimizer_attribute(temp_mod,"Method",2)
    set_optimizer_attribute(temp_mod,"Crossover",0)
    set_optimizer_attribute(temp_mod,"BarConvTol",opt_gap)
    # fix complicating variables
    for k in keys(sub.vars)
        fix_complicating_vars!(temp_mod,temp_sub,res,k)
    end
    # optimize
    optimize!(temp_mod)

    # collect results
    if termination_status(temp_mod) ∉ (MOI.OPTIMAL, MOI.LOCALLY_SOLVED)
        @error "Subproblem did not solve to optimality!"
        @info termination_status(temp_mod)
        return temp_mod
    end

    oracle_out = Oracle(temp_mod,temp_sub)
    return oracle_out

end

function add_cuts!(master::MasterProblem,oracle::Dict{Union{Int64,String},Oracle},res::Dict,iteration::Int64,settings::NamedTuple)    
    inv_cost = value(master.first_stage_obj)
    if settings.multicut_bool
        for year in settings.year_set
            tmp = Dict{Symbol,DataFrame}(k => leftjoin(oracle[year].duals[k], res[k], on = :symbol) for k in keys(master.vars))
            tmp_cut = @constraint(master.model,master.model[:θ][year]>=oracle[year].objVal - inv_cost + sum(sum(row[:dual]*(row[:master_var] - row[:value]) for row in eachrow(tmp[k])) for k in keys(master.vars)))
            set_name(tmp_cut,"cut_$(iteration)_$(year)")
            push!(master.cuts,[iteration,year,tmp_cut,iteration])
        end
    else
        tmp = Dict{Symbol,DataFrame}(k => leftjoin(oracle[year].duals[k], res[k], on = :symbol) for k in keys(master.vars))
        tmp_cut = @constraint(master.model,master.model[:θ]>=sum(oracle[year].objVal - inv_cost + sum(sum(row[:dual]*(row[:master_var] - row[:value]) for row in eachrow(tmp[k])) for k in keys(master.vars)) for year in settings.year_set))
        set_name(tmp_cut,"cut_$(iteration)")
        push!(master.cuts,[iteration,"agg",tmp_cut,iteration])
    end
end


function bundle_management(master::MasterProblem,settings::NamedTuple)
    # First update activity of cuts
    for row in eachrow(master.cuts)
        if dual(row[:cut]) > 0
            row[:last_active] = row[:iteration]
        end
    end
    # Then delete cuts that have not been active for a while if threshold has been set
    if !isempty(settings.cut_deletion_cutoff)
        inactive_cuts = filter(row->row[:iteration] - row[:last_active] > settings.cut_deletion_cutoff,master.cuts)
        master.cuts = setdiff(master.cuts,inactive_cuts)
        for row in eachrow(inactive_cuts)
            delete_constraint!(master.model,row[:cut])
        end
    end

    # Alternatively or additionally use maximum number of cuts
    if !isempty(settings.max_cut)
        if nrow(master.cuts) > settings.max_cut
            # sort cuts by activity
            sort!(master.cuts,order(:last_active,rev=true))
            # delete cuts
            inactive_cuts = master.cuts[settings.max_cut+1:end,:]
            master.cuts = master.cuts[1:settings.max_cut,:]
            for row in eachrow(inactive_cuts)
                delete_constraint!(master.model,row[:cut])
            end
        end
    end
end



##############################################################
## Testing and algorithm

# Paths etc
b=pwd()
datapath = b*"/data/test_model/"
include("benders_functions.jl")
#dtr = DieterModel(datapath; verbose=false);


# Settings
settings = (
    ts_path = "data/ts",
    year_set = [2012,2013,2014],
    maxhours = 720,
    lds_mode = true,
    h2_import_master = true,
    multicut_bool = true,
    probabilities = [1/3,1/3,1/3],
    cut_deletion_cutoff = 20,
    max_cut = 1000

)

# Define the master object
master = MasterProblem(Gurobi.Optimizer,datapath);

# Compile master problem
compile_master_problem!(master,settings)


# Create dictionary of subproblems
subs = Dict(year => SubProblem(year,Gurobi.Optimizer,datapath) for year in settings.year_set)

# Compile subproblems
map(year->build_sub_problem!(subs[year],settings,master.vars),settings.year_set)





oracle = Dict{Union{Int64,String},Oracle}(year => solve_subproblem(subs[year],res,0.01,settings) for year in settings.year_set)

