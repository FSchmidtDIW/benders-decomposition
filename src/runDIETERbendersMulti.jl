using DIETER
using Gurobi
using JuMP
using JuMP.Containers
using DataFrames
using CSV
using MathOptInterface

include("benders_functions.jl")

const env = Gurobi.Env()

# link to time series data to be used by subproblems
global ts_path = "data/ts/" # define where to get the data

# create DIETER object with all data
datapath = "data/test_model/"
dtr = DieterModel(datapath);

# Settings tuple
settings = (year_set = ["1996d1","1996d2","1996d3","1996d4"],
            lds_mode = true,
            init_lds_lvl = 0.5,
            year_prob = fill(1/4,4),
            maxhours = 24,
            solver = optimizer_with_attributes(() -> Gurobi.Optimizer(env), "OutputFlag" => 0),
            parameters = (
                max_iter = 1000,
                optimality_gap = 1e-3,
                stab_method = :qtr,
                cut_deletion_cutoff = 600,
                acc_tup = (range = [1e-2,1e-8],int = :log),
                descent_threshold = 0.0
            ),
            oda_kappa = 0.0,
            oda_count = 10,
            oda_orac_count = Inf,
            multicut_bool = true, # enable multi-cuts or not
            init_bool = true, # decide weather to compute initial stability_centre or not
            dual_subproblems = false,
            dual_num_control_bool = false,
            dual_num_control_eps = 1e-4,
            presolve_hours = 24,
            presolve_year = "1996d2",
            stab_params = Dict(
                :qtr => Dict(
                        :serious_step => Dict(),
                        :radius => Dict(1 => 5e-2),
                        :scaler => 1e-3,
                        :radius_scaler => 0.5,
                        :shrink_tol => 7.5e-4,
                        :low => 1e-3,
                        :step_limit => Inf
                ),
                :lvl => Dict(
                        :scaler => 1e-6,
                        :v_lvl => Dict{Int64,Float64}(),
                        :mu_max => 1.5,
                        :beta => 0.6,
                        :serious_step => Dict{Int64,Bool}(),
                        :mu => Dict{Int64,Float64}()
                ),
                :prx => Dict(
                        :serious_step => Dict(),
                        :prox_param => Dict(1 => 30.0),
                        :aux_term => Dict(),
                        :prox_aux => Dict(),
                        :prox_min => 10e-6,
                        :prox_a => 5.0,
                        :step_limit => 5,
                        :implementation => "Kiwiel1990",
                        :scaler => 1
                ),
                :dsb => Dict(
                        :serious_step => Dict(),
                        :prox_param => Dict(1 => 10.0),
                        :v_lvl => Dict{Int64,Float64}(),
                        :prox_a => 5, # used to divide prox_param if null step and proximal iteration
                        :prox_min => 10e-6,
                        :scaler => 1,
                        :mu_max => 3,
                        :beta => 0.5,
                        :mu => Dict{Int64,Float64}()
                )

            )
        );
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$


using Distributed
# Define number of workers
nb_workers = length(settings.year_set)
@static if Sys.islinux()
    using MatheClusterManagers.jl
    qsub(nb_workers,timelimit=345600,ram=16,mp=4) # replace 4 by NSLOTS
else
    addprocs(nb_workers; exeflags = "--project=.")
end

# Load packages and benders functions everywhere
@everywhere using DIETER, CSV, Gurobi, Distributed, ParallelDataTransfer
@everywhere include("benders_functions.jl")

# function to run sub-problems
function run_all_subproblems(year_set::Vector,res_obj,opt_tol,solver)
    solvedFut_dict = Dict{Int,Future}()
    for j in eachindex(year_set)
        solvedFut_dict[j] = @spawnat j+1 run_sub_distributed(res_obj,opt_tol)
    end
    return solvedFut_dict

end

function get_sub_results!(iteration::Int64,oracle_out::Dict,year_set::Vector,solvedFut_dict)
    for (k,v) in solvedFut_dict
        oracle_out[iteration][year_set[k]] = fetch(v)
    end
    return oracle_out
end




# begin model setup

t_pre_0 = time()

# Sense checks on settings
@assert length(settings.year_set) == length(settings.year_prob)
@assert sum(settings.year_prob) == 1
for x in settings.year_prob
    @assert 0<x<=1
end
# Create DenseAxisArray for probabilities
probs = DenseAxisArray(settings.year_prob,settings.year_set)


# Solver output configuration
#if settings.solver == Gurobi.Optimizer
#    env = Gurobi.Env()
#    solver= optimizer_with_attributes(() -> Gurobi.Optimizer(env), "OutputFlag" => 0)
#else
solver = settings.solver

#end

# Create master problem
mastr_dtr = deepcopy(dtr);
mstr = Model(solver);
set_silent(mstr);
define_model!(mstr,mastr_dtr,maxhours=0);
set_attribute(mstr,"OutputFlag",0);

# create container with complicating variables
mstr[:capa] = all_variables(mstr)
mstr[:investment_obj] = objective_function(mstr)

# add long-term storage complicating variables
if settings.lds_mode && DIETER.cond_ext(:h2,"all",mastr_dtr)
    # Run test if H2 module is active for at least one node
    # Define set of year steps
    step_set = [settings.year_set[s]*"_"*settings.year_set[s+1] for s in collect(1:(length(settings.year_set)-1))]
    # Add variables for all LDS technologies and all nodes and all steps
    @variable(mstr,LDSlev[n=mastr_dtr.sets[:n],s=mastr_dtr.sets[:stoh2],step=step_set; DIETER.cond_ext(:h2,n,mastr_dtr) && DIETER.cond_h2(s,n,mastr_dtr) && mastr_dtr.parameters[:h2_max_energy_sto][n,s] != 0] >= 0);
    # Add valid inequality to ensure consistency of capa and storage levels
    @constraint(mstr,LDSVI[n=mastr_dtr.sets[:n],s=mastr_dtr.sets[:stoh2],step=step_set; DIETER.cond_ext(:h2,n,mastr_dtr) && DIETER.cond_h2(s,n,mastr_dtr) && mastr_dtr.parameters[:h2_max_energy_sto][n,s] != 0],
        LDSlev[n,s,step] <= mstr[:H2_N_STO_E][n,s]
    );


mstr[:compl] = all_variables(mstr)
mstr[:lds_lvl] = setdiff(mstr[:compl],mstr[:capa])
end

# add value function model
if settings.multicut_bool
    @variable(mstr,θ[settings.year_set]>=0);
    @expression(mstr,obj_expr,mstr[:investment_obj] 
    + sum(probs[s]*mstr[:θ][s] for s in settings.year_set))
    @objective(mstr,Min,obj_expr);
else
    @variable(mstr,θ>=0);
    @expression(mstr,obj_expr,mstr[:investment_obj] 
    + mstr[:θ])
    @objective(mstr,Min,obj_expr)
end

# create subproblems
@info "Build subproblems in each worker"
sub_dtr = deepcopy(dtr)
year_set = settings.year_set
capa_vars = string.(mstr[:capa])
lds_lvl_vars = string.(mstr[:lds_lvl])
passobj(1,workers(),[:sub_dtr,:year_set,:solver,:datapath,:ts_path,:settings,:capa_vars,:lds_lvl_vars])




subTask_arr = map(workers()) do w
    t = @async @everywhere w begin
        # create subproblem
        function buildSub(id)
            tmp_dtr = create_sub_dict(sub_dtr,year_set[id])
            sub_m = Model(Gurobi.Optimizer)
            define_model!(sub_m,tmp_dtr,maxhours=settings.maxhours)
            return sub_m
        end
        const SUB_M = buildSub(myid()-1)

        function run_sub_distributed(res_data,opt_tol)
            tmp_oracle = solve_seq_subproblem(SUB_M,res_data,opt_tol,capa_vars,lds_lvl_vars;solver=solver)
            return tmp_oracle
        end

        return nothing
    end

    return w => t
end

subTask_arr = getindex.(values(subTask_arr),2)
if all(istaskdone.(subTask_arr))
    @info "All sub-problems are ready"
else
    @info "Waiting for subproblems to be ready"
    wait.(subTask_arr)
    @info "Sub-problems are ready"
end
# create tracker

active_cuts = []
#temp_duals = Array{Float64}(undef,settings.parameters.max_iter,length(settings.year_set)),
δ = Array{Float64}(undef,settings.parameters.max_iter,length(settings.year_set))
gap_container = []
lower_container = []
upper_container = []
time_container = []
stability_centre = []
oracle_output = Dict{Int64,Dict{Union{Int64,String},NamedTuple}}()

# initialize stability centre
if settings.init_bool
    init_mod = Model(solver)
    init_dtr = create_sub_dict(deepcopy(sub_dtr),settings.presolve_year)
    define_model!(init_mod,init_dtr,maxhours=settings.presolve_hours)
    optimize!(init_mod)
    x0 = value.(filter(x->(string(x) in string.(mstr[:capa])),all_variables(init_mod)))
    if settings.lds_mode
        sto_capa = value.(filter(x->contains(string.(x),"H2_N_STO_E"),all_variables(init_mod)))
        sto_capa_arr = Float64[]
        for el in sto_capa
            append!(sto_capa_arr,ones(length(step_set))*el*settings.init_lds_lvl)
        end
        append!(x0,sto_capa_arr)
    @assert length(x0) == length(mstr[:compl])
    else 
    @assert length(x0) == length(mstr[:capa])
    end
else
    x0 = lds_mode ? zeros(length(mstr[:compl])) : zeros(length(mstr[:capa]))
end



# initialize stability centre, lower bound uppper bound and
x_best = x0
push!(stability_centre,x_best)
lower_bound = 0
upper_bound = 1e14#Inf#2*objective_value(init_mod)
push!(upper_container,upper_bound)
push!(lower_container,lower_bound)
gap = 1.0
push!(gap_container,gap)
incumbency_count = 0
serious_count = 0
approx_count = 0
oracle_count = 0
if !isnothing(settings.parameters.stab_method)
    stab_params = settings[:stab_params][settings.parameters.stab_method]
    if settings.parameters.stab_method == :lvl || settings.parameters.stab_method == :dsb
        stab_params[:v_lvl][1] = (1-stab_params[:beta])*(upper_bound-lower_bound)
    end
end

@info "Time used for initialization $(round(time()-t_pre_0)) seconds."
push!(time_container,time()-t_pre_0)
# initialize time
t0 = time()

for k in 1:settings.parameters.max_iter
    if k==1 || mod(k,50) == 0
        println("Iteration  Lower Bound Upper Bound         Gap     Time")
    end

    # Step 1 Create and solve (stabilised) master problem
    if !isnothing(settings.parameters.stab_method)
        qm = add_stabilisation(k,settings.parameters.stab_method,stab_params,mstr,x_best,settings,upper_bound)
        set_silent(qm)
        set_optimizer_attribute(qm,"NumericFocus",3)
        optimize!(qm)
        while settings.parameters.stab_method == :lvl && termination_status(qm)  in (MOI.INFEASIBLE, MOI.INFEASIBLE_OR_UNBOUNDED)
            # If level set is empty set lower_bound to current level and reset descent target
            lower_bound = upper_bound - stab_params[:v_lvl][k]
            @info "Level set empty: Updating lower bound to $(lower_bound)"
            stab_params[:v_lvl][k] = (1-stab_params[:beta])*(upper_bound-lower_bound)
            # reformulate stabilised master problem with new level set constraint
            qm = add_stabilisation(k,settings.parameters.stab_method,stab_params,mstr,x_best,settings,upper_bound)
            # re-optimise
            optimize!(qm)
        end

        if termination_status(qm) ∉ (MOI.OPTIMAL, MOI.LOCALLY_SOLVED)
            @info "Stabilised master problem in iteration $k is infeasible, returning model object."
            return (model = qm, centre = x_best,oracle_output=oracle_output,stab_params), mstr
        end
        #return qm
        lower_bound_stabilised = objective_value(qm)
        # make sure that there are no numerical infeasibilities in candidate solution
        capa_res = settings.lds_mode ? distance_minimizer(mstr,value.(qm[:compl]),settings.solver) : distance_minimizer(mstr,value.(qm[:capa]),settings.solver)
        lds_lvl = settings.lds_mode ? Dict(qm[:lds_lvl] .=>value.(qm[:lds_lvl])) : nothing
        # compute lower bound based on unstabilised problem
        #if settings.parameters.stab_method != :lvl
            optimize!(mstr)
            lower_bound = objective_value(mstr)
        #else
        #end
    else
        optimize!(mstr)
        lower_bound = objective_value(mstr)
        capa_res = settings.lds_mode ? distance_minimizer(mstr,value.(qm[:compl]),settings.solver) : distance_minimizer(mstr,value.(qm[:capa]),settings.solver)
    end

    # Step 2 Pass on candidate iterate to sub-problems
    for k in keys(lds_lvl)
        if occursin(string(year)*"_",string(k))
            @constraint()
        elseif occursin("_"*string(year),string(k))
            println("start")
        else

        end
    end