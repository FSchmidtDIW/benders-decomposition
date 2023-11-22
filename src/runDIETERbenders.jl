using DIETER
using Gurobi
using JuMP
using JuMP.Containers
using DataFrames
using CSV
using Base.Threads
using MathOptInterface
using Dualization
using Printf
using ConvergencePlots

# import functions
include("benders_functions.jl")

const env = Gurobi.Env()
global ts_path = "../pre-processing/output/base/ts"

datapath = "data/test_model/"
dtr = DieterModel(datapath);

# Define run

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

# run Benders iteration
output_df, capacities, oracles = benders(dtr,settings) 


# take capacities and run submodels again
capacities



sort(oracles)



# test version for all storage technologies

# define set of year ends
step_def(x) = setdiff(x,[x[end]])
step_set = step_def(settings.year_set)
@variable(mstr,sto_lev[n=dtr.sets[:n],sto=dtr.sets[:sto],y=step_set; DIETER.cond_sto(sto,n, dtr)&& dtr.parameters[:sto_max_power_out][n,sto] != 0]>=0 )
@constraint(mstr,vi_sto_lev[n=dtr.sets[:n],sto=dtr.sets[:sto],y=step_set; DIETER.cond_sto(sto,n, dtr) && dtr.parameters[:sto_max_power_out][n,sto] != 0],
        mstr[:sto_lev][n,sto,y] <= mstr[:N_STO_E][n,sto]
);

mstr[:vi_sto_lev]

sub_m =  sub_mod_dict["1996d1"]

create_sub_models


function solve_subproblem_sequential(year::Union{String,Int64},sub_m::JuMP.AbstractModel,qm::JuMP.AbstractModel,,res,opt_gap;solver=Gurobi.Optimizer)
        temp_mod = copy(sub_m) # copy model so as to not alter the default subproblem
        set_silent(temp_mod)
        # define complicating variables
        capa = filter(x->contains(string.(x),"N_"),all_variables(temp_mod))

        if length(capa) != length(res)
            @error "Supplied iterate vector is not of the same length as complicating variables in subproblem!"
        end

        # fix complicating variables
        @constraint(temp_mod,fix[i=1:length(capa)],capa[i]==res[i])
    
        # fix storage start and ending levels
        if !isnothing(prev_year)
            @constraint(temp_mod,fix_start[n=dtr.sets[:n],sto=dtr.sets[:sto],y=prev_year],
                 temp_mod[:STO_L][n,sto,sub_dtr_dict[year].sets[:h][1]] == value(qm[:sto_lev][n,sto,prev_year])
                );
        end
    
        if !last_year
            @constraint(temp_mod,fix_end[n=dtr.sets[:n],sto=dtr.sets[:sto],y=year],
                temp_mod[:STO_L][n,sto,sub_dtr_dict[year].sets[:h][end]] == value(qm[:sto_lev][n,sto,year]))
        end 
    
        # solver settings (only Gurobi for now)
        set_optimizer(temp_mod,solver)
        set_optimizer_attribute(temp_mod,"Method",2)
        set_optimizer_attribute(temp_mod,"Crossover",0)
        set_optimizer_attribute(temp_mod,"BarConvTol",opt_gap)
        
        # Optimize
        optimize!(temp_mod)
        if termination_status(temp_mod)!=OPTIMAL
            @error "Subproblem did not solve to optimality!"
            return temp_mod
        end
        # Define output
        ret = (obj = objective_value(temp_mod),duals=dual.(temp_mod[:fix]),res_obj=objective_value(temp_mod)-dual.(temp_mod[:fix])'*res);
        return ret
    end

    sto_start[]

    year = "1996d4"
    temp_mod = sub_mod_dict[year]

    prev_year = findfirst(x -> x==year,settings.year_set) - 1 > 0 ? step_set[findfirst(x -> x==year,settings.year_set) - 1] : nothing
    last_year = !(year in step_set)

    if !isnothing(prev_year)
        @constraint(temp_mod,fix_start[n=dtr.sets[:n],sto=dtr.sets[:sto],y=prev_year], temp_mod[:STO_L][n,sto,sub_dtr_dict[year].sets[:h][1]] == value(qm[:sto_lev][n,sto,prev_year]))
    end

    if !last_year
        @constraint(temp_mod,fix_end[n=dtr.sets[:n],sto=dtr.sets[:sto],y=year], temp_mod[:STO_L][n,sto,sub_dtr_dict[year].sets[:h][end]] == value(qm[:sto_lev][n,sto,u]))
    end

    ret = (obj = objective_value(temp_mod),duals=dual.(temp_mod[:fix]))


        sub_mod_dict["1996d1"]
        sub_dtr_dict["1996d2"].sets[:h]
        dtr.sets[:h]
        temp_mod[:STO_L]["DE","Li-Ion",:]


    closed_dtr = create_sub_dict(deepcopy(dtr),1996)
    closed_mod = create_sub_models(1996,Dict(1996=>closed_dtr),96)
    optimize!(closed_mod)

    