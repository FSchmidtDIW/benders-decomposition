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

const env = Gurobi.Env()
global ts_path = "../pre-processing/output/base/ts"

datapath = "data/data_test_original/"
dtr = DieterModel(datapath);

settings = (year_set = [2014,2016],
            year_prob = [0.5,0.5],
            maxhours = 96,
            solver = optimizer_with_attributes(() -> Gurobi.Optimizer(env), "OutputFlag" => 0),
            parameters = (
                max_iter = 1000,
                optimality_gap = 1e-3,
                stab_method = :qtr,
                cut_deletion_cutoff = 600,
                acc_tup = (range = [1e-2,1e-8],int = :log),
                descent_threshold = 0.0
            ),
            multicut_bool = true, # enable multi-cuts or not
            init_bool = true, # decide weather to compute initial stability_centre or not
            dual_subproblems = true,
            dual_num_control_bool = false,
            dual_num_control_eps = 1e-4,
            presolve_hours = 24,
            presolve_year = 2014,
            stab_params = Dict(
                :qtr => Dict(
                        :serious_step => Dict(),
                        :radius => Dict(1 => 5e-2),
                        :scaler => 1e-3,
                        :radius_scaler => 0.5,
                        :shrink_tol => 7.5e-4,
                        :low => 1e-3,
                        :step_limit => 10
                ),
                :lvl => Dict(
                        :scaler => 1e-6,
                        :v_lvl => Dict{Int64,Float64}(),
                        :mu_max => 12,
                        :beta => 0.3,
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
                        :mu_max => 2,
                        :beta => 0.3,
                        :mu => Dict{Int64,Float64}()
                )

            )
        );

function compute_conv_tol(curr_opt_gap::Float64,conv_tol::Float64,meth_tup::NamedTuple)
    if meth_tup.int == :linear
        m = (meth_tup.range[1] - meth_tup.range[2])/(1-conv_tol)
        b = meth_tup.range[1] - m
        return b + m* curr_opt_gap
    elseif meth_tup.int == :exp
        m = log(meth_tup.range[1]/meth_tup.range[2])/(1-conv_tol)
        b = log(meth_tup.range[1]) - m
        return max(exp(b + m*curr_opt_gap),meth_tup.range[2])
    elseif meth_tup.int == :log
        b = meth_tup.range[1]
        m = (meth_tup.range[2] - meth_tup.range[1])/log(conv_tol)
        return max(b + m* log(curr_opt_gap),meth_tup.range[2])
    end
end


function solve_subproblem(sub_m::JuMP.AbstractModel,res,opt_gap;solver=Gurobi.Optimizer)
    temp_mod = copy(sub_m) # copy model so as to not alter the default subproblem
    set_silent(temp_mod)
    # define complicating variables
    capa = filter(x->contains(string.(x),"N_"),all_variables(temp_mod))
    if length(capa) != length(res)
        @error "Supplied iterate vector is not of the same length as complicating variables in subproblem!"
    end
    # fix complicating variables
    @constraint(temp_mod,fix[i=1:length(capa)],capa[i]==res[i])

    # deduct investment costs from 

    # solver settings (only Gurobi for now)
    set_optimizer(temp_mod,solver)
    set_optimizer_attribute(temp_mod,"Method",2)
    set_optimizer_attribute(temp_mod,"Crossover",0)
    set_optimizer_attribute(temp_mod,"BarConvTol",opt_gap)
    
    # Optimize
    @time optimize!(temp_mod)
    if termination_status(temp_mod)!=OPTIMAL
        @error "Subproblem did not solve to optimality!"
        return temp_mod
    end
    # Define output
    ret = (obj = objective_value(temp_mod),duals=dual.(temp_mod[:fix]),res_obj=objective_value(temp_mod)-dual.(temp_mod[:fix])'*res);
    return ret
end

function solve_oda_subproblem(sub_m::JuMP.AbstractModel,res,oracle_out::Dict,target::Float64,year,opt_gap;solver=Gurobi.Optimizer)
    if !isempty(oracle_out)
        approx_vec = []
        approx_idx = []
        for y in settings.year_set
            approx_tmp, idx = findmax([oracle_out[j][y].res_obj + oracle_out[j][y].duals'*res for j in keys(oracle_out)])
            push!(approx_vec,approx_tmp)
            push!(approx_idx,idx)
        end
        approx_f = sum(settings.year_prob[i]*approx_vec[i] for i in eachindex(approx_vec))
        approx_vec = DenseAxisArray(approx_vec,settings.year_set)
        approx_idx = DenseAxisArray(approx_idx,settings.year_set)
        if approx_f > target
            tmp = (obj = approx_vec[year],duals=oracle_out[approx_idx[year]][year].duals,res_obj = approx_f - oracle_out[approx_idx[year]][year].duals'*res)
            @info "The approximate model value is equal to  : $approx_f"
            return tmp, false
        else
            ret = solve_subproblem(sub_m,res,opt_gap;solver)
            return ret, true
        end
    else
        ret = solve_subproblem(sub_m,res,opt_gap;solver)
        return ret, true
    end
end

        





function solve_dual_subproblem(sub_m::JuMP.AbstractModel,res,opt_gap;solver=Gurobi.Optimizer)
    temp_mod = copy(sub_m) # copy model so as to not alter the default subproblem
    set_silent(temp_mod)
    # define complicating variables
    temp_mod[:fix] = filter(x->contains(string.(x),"dual_fix"),all_variables(temp_mod))
    if length(temp_mod[:fix]) != length(res)
        @error "Supplied iterate vector is not of the same length as complicating variables in subproblem!"
    end
    # fix complicating variables
    for i in eachindex(temp_mod[:fix])
        set_objective_coefficient(temp_mod,temp_mod[:fix][i],res[i])
    end

    # solver settings (only Gurobi for now)
    set_optimizer(temp_mod,solver)
    set_optimizer_attribute(temp_mod,"Method",2)
    set_optimizer_attribute(temp_mod,"Crossover",0)
    set_optimizer_attribute(temp_mod,"BarConvTol",opt_gap)
    #set_optimizer_attribute(temp_mod,"NumericFocus",2)
    # Optimize
    @time optimize!(temp_mod)
    if termination_status(temp_mod) ∉ (OPTIMAL, LOCALLY_SOLVED)
        @error "Subproblem did not solve to optimality!"
        return temp_mod
    end
    # Define output
    ret = (obj = objective_value(temp_mod),duals=value.(temp_mod[:fix]));
    return ret
end

function solve_oda_dual_subproblem(sub_m::JuMP.AbstractModel,res,opt_gap;solver=Gurobi.Optimizer)
    if !isempty(oracle_out)
        temp_mod = copy(sub_m)
        temp_mod[:fix] = filter(x->contains(string.(x),"dual_fix"),all_variables(temp_mod))

            # fix complicating variables
        for i in eachindex(temp_mod[:fix])
            set_objective_coefficient(temp_mod,temp_mod[:fix][i],res[i])
        end
        obj = objective_function(temp_mod)
        tmp_vec = []
        for j in keys(oracle_out)
            point = Dict(all_variables(temp_mod) .=> oracle_out[j][year].values)
            push!(tmp_vec, value(z->point[z],f))
        end
        obj,findmax(tmp_vec)
    end
end

function distance_minimizer(top_m::JuMP.AbstractModel,res,solver)
    m = copy(top_m) # top_m must not include any cuts
    set_optimizer(m,solver)    
    set_silent(m)
    # Add constraints to minimize L1 distance between master problem feasible space and current solution 
    all_vars = all_variables(m)
    @expression(m,x[i=eachindex(res)],all_vars[i]-res[i])
    @variable(m,helper[i=eachindex(res)])
    @constraint(m,[i=eachindex(res)],helper[i]==x[i])
    @variable(m,t)
    @objective(m,Min,t)
    @constraint(m,[t;helper] in MOI.NormOneCone(1+length(res)))

    # optimize
    optimize!(m)
    return value.(m[:capa])
end



function add_stabilisation(iter::Int64,method::Val{:qtr},stab_params::Dict,top_m::JuMP.Model,centre::Vector{Float64},settings::NamedTuple,upper_bound::Float64)
    k = iter
    stab_top_m = copy(top_m)
    @assert length(centre) == length(top_m[:capa])
    set_optimizer(stab_top_m,settings.solver)
    set_silent(stab_top_m)
    l2_norm = @expression(stab_top_m,sum((stab_top_m[:capa][i] - centre[i])^2 for i in eachindex(centre)))
    orig_dist = @expression(stab_top_m,sum(centre[i] for i in eachindex(centre))^2)
    # add quadratic trust region
    @constraint(stab_top_m,qtr,
        stab_params[:scaler]*l2_norm <= stab_params[:scaler]*stab_params[:radius][k]^2*orig_dist
    );

    return stab_top_m
end

function add_stabilisation(iter::Int64,method::Val{:lvl},stab_params::Dict,top_m::JuMP.Model,centre::Vector{Float64},settings::NamedTuple,upper_bound::Float64)
    k = iter
    v = stab_params[:v_lvl][k]
    ℓ = upper_bound - v
    stab_top_m = copy(top_m)
    @assert length(centre) == length(top_m[:capa])
    set_optimizer(stab_top_m,settings.solver)
    set_silent(stab_top_m)
    l2_norm = @expression(stab_top_m,sum((stab_top_m[:capa][i] - centre[i])^2 for i in eachindex(centre)))
    
    # create level model
    @objective(stab_top_m,Min,0.5*l2_norm*stab_params[:scaler])
    @constraint(stab_top_m,lvl_set, stab_top_m[:obj_expr] <= ℓ);
    
    return stab_top_m
end


function add_stabilisation(iter::Int64,method::Val{:prx},stab_params::Dict,top_m::JuMP.Model,centre::Vector{Float64},settings::NamedTuple,upper_bound::Float64)
    k = iter
    τ = stab_params[:prox_param][k]
    stab_top_m = copy(top_m)
    @assert length(centre) == length(top_m[:capa])
    set_optimizer(stab_top_m,settings.solver)
    set_silent(stab_top_m)
    l2_norm = @expression(stab_top_m,sum((stab_top_m[:capa][i] - centre[i])^2 for i in eachindex(centre)))
    
    # create proximal model
    @objective(stab_top_m,Min,stab_top_m[:obj_expr] + 0.5*(1/τ)*l2_norm*stab_params[:scaler])

    return stab_top_m
end

function add_stabilisation(iter::Int64,method::Val{:dsb},stab_params::Dict,top_m::JuMP.Model,centre::Vector{Float64},settings::NamedTuple,upper_bound::Float64)
    k = iter
    v = stab_params[:v_lvl][k]
    ℓ = upper_bound - v
    τ = stab_params[:prox_param][k]
    stab_top_m = copy(top_m)
    @assert length(centre) == length(top_m[:capa])
    set_optimizer(stab_top_m,settings.solver)
    set_silent(stab_top_m)
    l2_norm = @expression(stab_top_m,sum((stab_top_m[:capa][i] - centre[i])^2 for i in eachindex(centre)))
    
    # create proximal model
    @objective(stab_top_m,Min,stab_top_m[:obj_expr] + 0.5*(1/τ)*l2_norm*stab_params[:scaler])
    @constraint(stab_top_m,lvl_set,stab_top_m[:obj_expr]<= ℓ)

    return stab_top_m
end

add_stabilisation(iter::Int64,method::Symbol,stab_params::Dict,top_m::JuMP.Model,centre::Vector{Float64},settings::NamedTuple,upper_bound::Float64) = add_stabilisation(iter::Int64,Val{method}(),stab_params::Dict,top_m::JuMP.Model,centre::Vector{Float64},settings::NamedTuple,upper_bound::Float64)

function dynamic_par_adjustment!(iter::Int64,method::Val{:qtr},stab_params::Dict,upper_bound,lower_bound,lower_bound_stabilised,incumbency_count,serious_count)
    k = iter
    # implement Benders paper rule for qtr
    if abs(1 - lower_bound/lower_bound_stabilised) < stab_params[:shrink_tol] && stab_params[:radius][k] > stab_params[:low]
        @info "Halving trust region."
        stab_params[:radius][k+1] = max(stab_params[:radius_scaler]*stab_params[:radius][k],stab_params[:low])
    elseif incumbency_count > stab_params[:step_limit]
        @info "Doubling trust region"
        stab_params[:radius][k+1] = stab_params[:radius][k]*(1/stab_params[:radius_scaler])
        incumbency_count = 0
    else
        stab_params[:radius][k+1] = stab_params[:radius][k]
    end
    return stab_params
end

function dynamic_par_adjustment!(iter::Int64,method::Val{:prx},stab_params::Dict,upper_bound,lower_bound,lower_bound_stabilised,incumbency_count,serious_count)
    # implement proximal bundle method from de Oliveira, W., Solodov, M.: A doubly stabilized bundle method for nonsmooth convex optimization, Math. Program., Ser. A (2016) 156:125–159
    k = iter
    if stab_params[:implementation] == "Kiwiel1990" 
        stab_params[:prox_aux][k] = 2*stab_params[:prox_param][k]*(1+stab_params[:aux_term][k])
    elseif stab_params[:implementation] == "Lemarechal1995"
        # add later
    end

    if stab_params[:serious_step][k]
        if serious_count > stab_params[:step_limit]
            stab_params[:prox_aux][k] = stab_params[:prox_a]*stab_params[:prox_aux][k]
        end
        stab_params[:prox_param][k+1] = min(stab_params[:prox_aux][k],10*stab_params[:prox_param][k])
    else
        stab_params[:prox_param][k+1] = min(stab_params[:prox_param][k],max(stab_params[:prox_aux][k],stab_params[:prox_param][k]/stab_params[:prox_a],stab_params[:prox_min]))
    end
    return stab_params
end

function dynamic_par_adjustment!(iter::Int64,method::Val{:lvl},stab_params::Dict,upper_bound,lower_bound,lower_bound_stabilised,incumbency_count,serious_count)
    # Implement level bundle method
    k = iter
    if stab_params[:serious_step][k]
        stab_params[:v_lvl][k+1] = min(stab_params[:v_lvl][k],(1-stab_params[:beta])*(upper_bound-lower_bound))
    else 
        if stab_params[:mu][k] > stab_params[:mu_max]
            @info "μ-test passed: Descreasing step."
            stab_params[:v_lvl][k+1] = stab_params[:beta]*stab_params[:v_lvl][k]
        else
            stab_params[:v_lvl][k+1] = stab_params[:v_lvl][k]
        end
    end
    return stab_params
end

function dynamic_par_adjustment!(iter::Int64,method::Val{:dsb},stab_params::Dict,upper_bound,lower_bound,lower_bound_stabilised,incumbency_count,serious_count)
    # Implement level bundle method
    k = iter
    if stab_params[:serious_step][k]
        stab_params[:v_lvl][k+1] = min(stab_params[:v_lvl][k],(1-stab_params[:beta])*(upper_bound-lower_bound))
        stab_params[:prox_param][k+1] = stab_params[:prox_a]*stab_params[:mu][k]*stab_params[:prox_param][k]
    else 
        if stab_params[:mu][k] > stab_params[:mu_max]
            stab_params[:v_lvl][k+1] = stab_params[:beta]*stab_params[:v_lvl][k]
        else
            stab_params[:v_lvl][k+1] = stab_params[:v_lvl][k]
        end
        stab_params[:prox_param][k+1] = stab_params[:prox_param][k]/stab_params[:prox_a]
    end
    return stab_params
end


dynamic_par_adjustment!(iter::Int64,method::Symbol,stab_params::Dict,upper_bound,lower_bound,lower_bound_stabilised,incumbency_count,serious_count) = dynamic_par_adjustment!(iter::Int64,Val{method}(),stab_params::Dict,upper_bound,lower_bound,lower_bound_stabilised,incumbency_count,serious_count) 

function benders(dtr::DieterModel,
                settings::NamedTuple)


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

    if settings.dual_subproblems
        # Define subproblems as dual problems
        sub_dtr = deepcopy(dtr)
        sub_dtr_dict = Dict()
        sub_mod_dict = Dict()
        sub_m = Model()
        for y in settings.year_set
            sub_dtr_dict[y] = create_sub_dict(sub_dtr,y)
            tmp = copy(sub_m)
            define_model!(tmp,sub_dtr_dict[y],maxhours=settings.maxhours)
            tmp[:capa] = filter(x->contains(string.(x),"N_"),all_variables(tmp))
            @constraint(tmp,fix[i=eachindex(tmp[:capa])],tmp[:capa][i]==1)
            d_tmp = dualize(tmp,dual_names=DualNames("dual_",""))
            sub_mod_dict[y] = d_tmp
        end
    else   
        # Define subproblems
        sub_dtr = deepcopy(dtr)
        sub_dtr_dict = Dict()
        sub_mod_dict = Dict()
        sub_m = Model()
        for y in settings.year_set
            sub_dtr_dict[y] = create_sub_dict(sub_dtr,y)
            tmp = copy(sub_m)
            define_model!(tmp,sub_dtr_dict[y],maxhours=settings.maxhours)
            sub_mod_dict[y] = tmp
        end
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
    oracle_output = Dict{Int64,Dict{Int64,NamedTuple}}()
    
    # initialize stability centre
    if settings.init_bool
        init_mod = Model(solver)
        define_model!(init_mod,sub_dtr_dict[settings.presolve_year],maxhours=settings.presolve_hours)
        optimize!(init_mod)
        x0 = value.(filter(x->contains(string.(x),"N_"),all_variables(init_mod)))
        @assert length(x0) == length(mstr[:capa])
    else
        x0 = zeros(length(mstr[:capa]))
    end

    


    # initialize stability centre, lower bound uppper bound and
    x_best = x0
    push!(stability_centre,x_best)
    lower_bound = 0
    upper_bound = 1e12#objective_value(init_mod)
    push!(upper_container,upper_bound)
    push!(lower_container,lower_bound)
    gap = 1.0
    push!(gap_container,gap)
    incumbency_count = 0
    serious_count = 0
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
    
    #println("Iteration  Lower Bound Upper Bound         Gap")
    for k in 1:settings.parameters.max_iter
        if k==1 || mod(k,50) == 0
            println("Iteration  Lower Bound Upper Bound         Gap     Time")
        end

        # create and solve (stabilised) master problem
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
            res = distance_minimizer(mstr,value.(qm[:capa]),settings.solver)
            # compute lower bound based on unstabilised problem
            #if settings.parameters.stab_method != :lvl
                optimize!(mstr)
                lower_bound = objective_value(mstr)
            #else
            #end
        else
            optimize!(mstr)
            lower_bound = objective_value(mstr)
            res = distance_minimizer(mstr,value.(mstr[:capa]),settings.solver)
        end


        # Call oracle
        tmp_dict = Dict{Int64,NamedTuple}()
        con_exact_bool = []
        # This bit will be parallelised
        @time for year in settings.year_set
            tmp_dict[year] = settings.dual_subproblems ? solve_dual_subproblem(sub_mod_dict[year],res,compute_conv_tol(gap,settings.parameters.optimality_gap,settings.parameters.acc_tup);solver) : solve_subproblem(sub_mod_dict[year],res,compute_conv_tol(gap,settings.parameters.optimality_gap,settings.parameters.acc_tup);solver)
            # solve_oda_subproblem(sub_mod_dict[year],res,oracle_output,0.5*upper_bound+0.5*lower_bound,year,compute_conv_tol(gap,settings.parameters.optimality_gap,settings.parameters.acc_tup);settings.solver)
            #push!(con_exact_bool,exact_bool)
        end
        #exact_bool = con_exact_bool[1]
        # save output
        oracle_output[k] = tmp_dict

        

        # Bundle management
        
            # identify near-zero duals and round off to avoid numerical issues
        if settings.dual_num_control_bool
            for year in settings.year_set
                oracle_output[k][year].duals[oracle_output[k][year].duals[abs.(oracle_output[k][year].duals) .< settings.dual_num_control_eps].= zero(eltype(oracle_output[k][year].duals))] 
            end
        end
        
        if settings.multicut_bool
            if k>1
                for s in settings.year_set push!(active_cuts,(k-1,s)) end
                temp_duals = DenseAxisArray([dual(constraint_by_name(mstr,"cut$(j)[$s]")) for (j,s) in active_cuts],active_cuts)
                temp_cuts = active_cuts
                for (j,s) in active_cuts
                    idx = findfirst(x->x==s,settings.year_set)
                    if temp_duals[(j,s)]>0
                        δ[j,idx] = k
                    end
                    if (k - δ[j,idx] > settings.parameters.cut_deletion_cutoff)&!isnothing(constraint_by_name(mstr,"cut$j[$s]"))
                        JuMP.delete(mstr,constraint_by_name(mstr,"cut$j[$s]"))
                        deleteat!(temp_cuts,findall(x->x==(j,s),temp_cuts)[1])
                        @info "Deleting cut $(j)[$s]"
                    end
                end
                active_cuts = temp_cuts
            end
        else
            if k>1
                push!(active_cuts,k-1)
                temp_duals = DenseAxisArray([dual(constraint_by_name(mstr,"cut$(j)")) for j in active_cuts],active_cuts)
                temp_cuts = active_cuts
                for j in active_cuts
                    if temp_duals[j] >0
                        δ[j] = k
                    end
                   if ((k - δ[j]) > settings.parameters.cut_deletion_cutoff)&!isnothing(constraint_by_name(mstr,"cut$j"))
                        JuMP.delete(mstr,constraint_by_name(mstr,"cut$j"))
                        temp_cuts = setdiff(temp_cuts,j)
                       @info "Deleting cut $(j)"
                    end
                end
                active_cuts = temp_cuts
            end
        end


        # Update cutting plane model
        if settings.multicut_bool
            for s in settings.year_set
                cut = @constraint(mstr,
                                mstr[:θ][s] >= oracle_output[k][s].obj - mstr[:investment_obj] + sum(oracle_output[k][s].duals[i]*(mstr[:capa][i] - res[i] ) for i in eachindex(mstr[:capa]))
                            );
            set_name(cut,"cut$(k)[$s]")
            end
        else
            cut = @constraint(mstr,
                                mstr[:θ] >= sum(probs[s]*(oracle_output[k][s].obj - mstr[:investment_obj] + sum(oracle_output[k][s].duals[i]*(mstr[:capa][i] - res[i] ) for i in eachindex(mstr[:capa]))) for s in settings.year_set)
                            );
            set_name(cut,"cut$(k)")
        end

        # compute oracle objective value and expected descent
        v = upper_bound - lower_bound
        f = sum(probs[s]*oracle_output[k][s].obj for s in settings.year_set)
        if settings.parameters.stab_method == :prx
            improv = upper_bound - f
            aux = v/improv
            stab_params[:aux_term][k] = aux
        end

        adjCtr_bool = false
        # serious step test
        if upper_bound - settings.parameters.descent_threshold*v > f #&& exact_bool
            upper_bound = f
            x_best = res
            push!(stability_centre,x_best)
            #@info "Updating stability centre"
            adjCtr_bool = true
        end


        # dynamic adjustment of stabilisation parameters
        if !isnothing(settings.parameters.stab_method)
            stab_params[:serious_step][k] = adjCtr_bool
            incumbency_count = adjCtr_bool ? 0 : incumbency_count + 1 
            serious_count = adjCtr_bool ? serious_count + 1 : 0
            if settings.parameters.stab_method == :lvl || settings.parameters.stab_method == :dsb
                stab_params[:mu][k] = 1-(dual(qm[:lvl_set])/stab_params[:scaler])
            end
            stab_params = dynamic_par_adjustment!(k,settings.parameters.stab_method,stab_params,upper_bound,lower_bound,lower_bound_stabilised,incumbency_count,serious_count)
            if settings.parameters.stab_method == :qtr
                if stab_params[:radius][k+1] - stab_params[:radius][k] > 0
                    incumbency_count = 0
                end
            elseif settings.parameters.stab_method == :prx
                if serious_count > stab_params[:step_limit]
                    serious_count = 0
                end
            end
        end
            #return stab_params
        # check convergence
        gap = (upper_bound-lower_bound)/upper_bound
        
        # record iteration
        star = adjCtr_bool ? "*" : ""
        if settings.parameters.stab_method == :lvl
            print_iteration(k,star,lower_bound,upper_bound,gap,time()-t0,stab_params[:mu][k])
        else
            print_iteration(k,star,lower_bound,upper_bound,gap,time()-t0)
        end
        push!(gap_container,gap)
        push!(time_container,time()-t0)
        push!(lower_container,lower_bound)
        push!(upper_container,upper_bound)

        if gap < settings.parameters.optimality_gap
            @info "Terminating with optimal solution"
            break
        end
        
    
    end

    if settings.parameters.stab_method == :qtr
        dynPar = [stab_params[:radius][i] for i in 1:length(gap_container)]
    elseif settings.parameters.stab_method == :prx
        dynPar = [stab_params[:prox_param][i] for i in 1:length(gap_container)]
    elseif settings.parameters.stab_method == :lvl 
        dynPar = [stab_params[:v_lvl][i] for i in 1:length(gap_container)]
    elseif settings.parameters.stab_method == :dsb
        dynPar = [(stab_params[:v_lvl][i],stab_params[:prox_param][i]) for i in 1:length(gap_container)]
    end

    results_df = DataFrame(
        :iteration => collect(1:length(gap_container)),
        :lower_bound => lower_container,
        :upper_bound => upper_container,
        :gap => gap_container,
        :time => time_container,
        :dynPar => dynPar
    )

    return results_df, x_best, oracle_output
    #(upper_bound = upper_bound,
   #lower_bound = lower_bound,
    #capacities = x_best,
    #gap = gap_container,
    #time = time_container,
    #stab_params = stab_params,
    #oracle_output = oracle_output)
        


end


function print_iteration(k,star::String, args...)
    f(x) = Printf.@sprintf("%12.4e", x)
    println(lpad(k, 9), " ", join(f.(args), " "),star)
    return
end



function create_sub_dict(sub_dtr::DieterModel,year::Int64)
    tmp_dtr = sub_dtr
    tmp_avail = CSV.read(joinpath(ts_path,"$(year)_availability.csv"),DataFrame)
    tmp_load = CSV.read(joinpath(ts_path,"$(year)_load.csv"),DataFrame)
    for n in tmp_dtr.sets[:n], r in setdiff(tmp_dtr.sets[:nondis],["ror"])
        tmp_dtr.parameters[:gen_availability][n,r,:] = filter(row->row.n == "$(n)",tmp_avail)[:,Symbol("$(r)")]
        tmp_dtr.parameters[:node_electricity_demand][n,:] = filter(row->row.n=="$(n)",tmp_load)[:,:load]
    end

    #sub_dtr_dict[year] = 
    return tmp_dtr

end

function create_sub_models(year::Int,sub_dtr_dict::Dict,maxhours)
    tmp_dtr = sub_dtr_dict[year]
    tp = JuMP.Model(Gurobi.Optimizer)
    define_model!(tp,tmp_dtr,maxhours=maxhours)
   return tp
end


#closed_dtr = create_sub_dict(dtr,2012)
#closed_m = Model(Gurobi.Optimizer)
#define_model!(closed_m,closed_dtr,maxhours=96)
#optimize!(closed_m)

qm, x_best, oracles = benders(dtr,settings)

qm
    termination_status(qm)
qm[:qtr] 

x_best |> optimize!
termination_status(x_best)
objective_value(x_best)
constraint_by_name(x_best,"cut300[2012]")

(objective_value(closed_m) - (qm[!,:upper_bound] |> minimum))/objective_value(closed_m)

solve_subproblem(closed_m,x_best,0.001)

qm |> CSV.write("dsb_output20142016.csv")

optimize!(qm.model)


optimize!(qm.model)
termination_status(qm.model)
find_infeasible_constraints(qm.model)
objective_function(qm.model)
objective_value(qm.model)


solve_subproblem(closed_m,qm.capacities,1e-8)


qm.model[:qtr]

delete(qm.model,constraint_by_name(qm.model,"cut21"))
unregister(qm.model,constraint_by_name(qm.model,"cut21"))



value.(qm.model[:N_TECH]["DE",:])
value.(closed_m[:N_TECH]["DE",:])
dtr.parameters[:gen_max_power]["DE",:]


(qm.oracle_output[479][2004].obj - objective_value(closed_m))/qm.oracle_output[479][2004].obj


m = direct_model(Gurobi.Optimizer())
define_model!(m,dtr,maxhours=336)
set_optimizer_attribute(m,"Method",2)
set_optimizer_attribute(m,"Crossover",0)




c = []
thr = 10^11
obj = Float64[]
function absolute_stopping_criterion(cb_data,cb_where)
    objbound =Ref{Cdouble}()
    obj_primal = Ref{Cdouble}()
    Gurobi.GRBcbget(cb_data,cb_where,Gurobi.GRB_CB_BARRIER_DUALOBJ,objbound)
    objbound = objbound[]
    push!(c,objbound)
    
    if maximum(c) > thr
        Gurobi.GRBcbget(cb_data,cb_where,Gurobi.GRB_CB_BARRIER_PRIMOBJ,obj_primal)
        
        obj_primal = obj_primal[]
        push!(obj,obj_primal)
        #push!(obj,obj_primal)
        Gurobi.load_callback_variable_primal(cb_data,cb_where)
        callback_value(cb_data,)
        Gurobi.GRBterminate(backend(m).inner)
    end
    return obj 
    #@show((maximum(container)))
    #max = maximum(container)
    #if max > f_best
    #    @info max
    #end
end

obj

MOI.set(m,Gurobi.CallbackFunction(),absolute_stopping_criterion)
#GC.enable(false)
optimize!(m)


x_exact = solve_subproblem(sub_mod_dict[2014],res,1e-8;solver=settings.solver)
x_inexact = solve_subproblem(sub_mod_dict[2014],res,0.01;solver=settings.solver)
x_exact.obj-x_inexact.obj


d = copy(sub_mod_dict[2014])
d[:capa] = filter(x->contains(string.(x),"N_"),all_variables(temp_mod))
set_optimizer(d,Gurobi.Optimizer)
optimize!(d)
res = value.(d[:capa])


c

termination_status(d)

filter(x->contains(string.(x),"cut"),all_constraints(qm.model,include_variable_in_set_constraints=false))[1] |> print

x
temp_mod = 0
temp_mod = copy(sub_mod_dict[2014]) # copy model so as to not alter the default subproblem
#set_silent(temp_mod)
# define complicating variables
capa = filter(x->contains(string.(x),"N_"),all_variables(temp_mod))
#if length(capa) != length(res)
#    @error "Supplied iterate vector is not of the same length as complicating variables in subproblem!"
#end
# fix complicating variables
@constraint(temp_mod,fix[i=1:length(capa)],capa[i]==x_best[i])

# deduct investment costs from 

# solver settings (only Gurobi for now)
set_optimizer(temp_mod,Gurobi.Optimizer)
set_optimizer_attribute(temp_mod,"Method",2)
set_optimizer_attribute(temp_mod,"Crossover",0)
#set_optimizer_attribute(temp_mod,"BarConvTol",opt_gap)

# Optimize
optimize!(temp_mod)
if termination_status(temp_mod)!=OPTIMAL
    @error "Subproblem did not solve to optimality!"
    return temp_mod
end
# Define output

d = dualize(temp_mod,dual_names=DualNames("dual_","dual_"));
d[:fixed] = filter(x->contains(string.(x),"dual_fix"),all_variables(d))
set_optimizer(d,Gurobi.Optimizer)
set_optimizer_attribute(d,"Method",2)
set_optimizer_attribute(d,"Crossover",0)

c = []
thr = 1e11
obj = Float64[]
fix_out = []
function absolute_stopping_criterion(cb_data,cb_where)
    objbound =Ref{Cdouble}()
    obj_primal = Ref{Cdouble}()
    Gurobi.GRBcbget(cb_data,cb_where,Gurobi.GRB_CB_BARRIER_PRIMOBJ,objbound)
    Gurobi.GRBcbget(cb_data,cb_where,Gurobi.GRB_CB_BARRIER_DUALOBJ,obj_primal)
    objbound = objbound[]
    obj_primal = obj_primal[]
    push!(obj,obj_primal)
    push!(c,objbound)
    if maximum(c) > thr
        #push!(obj,obj_primal)
        #Gurobi.load_callback_variable_primal(cb_data,cb_where)
        #fixed = callback_value.(cb_data,fixed)
        #push!(fix_out,fixed)
        #GRBterminate(backend(d).optimizer.model.inner)
        throw(Gurobi.CallbackAbort())
        
    end
    return obj 
    #@show((maximum(container)))
    #max = maximum(container)
    #if max > f_best
    #    @info max
    #end
end


MOI.set(d,Gurobi.CallbackFunction(),absolute_stopping_criterion)
#set_optimizer_attribute(d,"BarConvTol",1)
set_optimizer_attribute(d,"Method",2)
set_optimizer_attribute(d,"Crossover",0)

#GC.enable(false)
optimize!(d)
termination_status(d)
objective_value(d)

DataFrame(:primal => obj) |> CSV.write("primal_out.csv")
filter(x->(x!=0),round.(obj)) |> minimum


m = Model(Gurobi.Optimizer)
define_model!(m,dtr,maxhours=336)
set_optimizer_attribute(m,"Method",2)
set_optimizer_attribute(m,"Crossover",0)
#set_optimizer_attribute(m,"BarConvTol",0.1)
optimize!(m)
(objective_value(m) - 1.46240846e+11)/objective_value(m)
dual_objective_value(m)


temp_mod = copy(sub_mod_dict[2014])
set_optimizer
capa = filter(x->contains(string.(x),"N_"),all_variables(temp_mod))
    if length(capa) != length(res)
        @error "Supplied iterate vector is not of the same length as complicating variables in subproblem!"
    end
    # fix complicating variables
    @constraint(temp_mod,fix[i=1:length(capa)],capa[i]==res[i])


d_temp_mod = dualize(temp_mod,dual_names=DualNames("dual_","dual_"));
optimize!(temp_mod)

objective_function(d_temp_mod)
set_optimizer(d_temp_mod,Gurobi.Optimizer)
set_optimizer_attribute(d_temp_mod,"Method",2)
optimize!(d_temp_mod)
set_optimizer_attribute(d_temp_mod,"Crossover",0)
objective_value(d_temp_mod)

set_optimizer(temp_mod,Gurobi.Optimizer)
set_optimizer_attribute(temp_mod,"Method",2)
set_optimizer_attribute(temp_mod,"Crossover",0)
optimize!(temp_mod)
objective_value(temp_mod) - objective_value(d_temp_mod)

objective_function(d_temp_mod) |> println
write_to_file(d_temp_mod,"dual_subproblem.lp")

d_temp_mod[:dual_capa] = filter(x->contains(string.(x),"dual_fix"),all_variables(d_temp_mod))
for i in eachindex(d_temp_mod[:dual_capa])
    set_objective_coefficient(d_temp_mod,d_temp_mod[:dual_capa][i],0)
end


sub_mod_dict

res = x0
temp_mod = copy(sub_mod_dict[2014]); # copy model so as to not alter the default subproblem
# define complicating variables
temp_mod[:fix] = filter(x->contains(string.(x),"dual_fix"),all_variables(temp_mod))
if length(temp_mod[:fix]) != length(res)
    @error "Supplied iterate vector is not of the same length as complicating variables in subproblem!"
end
# fix complicating variables
@time for i in eachindex(temp_mod[:fix])
    set_objective_coefficient(temp_mod,temp_mod[:fix][i],res[i])
end

# solver settings (only Gurobi for now)
set_optimizer(temp_mod,solver)
set_optimizer_attribute(temp_mod,"Method",2)
set_optimizer_attribute(temp_mod,"Crossover",0)
set_optimizer_attribute(temp_mod,"BarConvTol",1e-8)
#set_optimizer_attribute(temp_mod,"NumericFocus",2)
# Optimize
@time optimize!(temp_mod)
if termination_status(temp_mod) ∉ (OPTIMAL, LOCALLY_SOLVED)
    @error "Subproblem did not solve to optimality!"
    return temp_mod
end
# Define output
ret = (obj = objective_value(temp_mod),duals=value.(temp_mod[:fix]));

solution_summary(temp_mod)


temp_mod2= copy(sub_mod_dict[2014]) # copy model so as to not alter the default subproblem
set_silent(temp_mod2)
# define complicating variables
capa = filter(x->contains(string.(x),"N_"),all_variables(temp_mod2))
if length(capa) != length(res)
    @error "Supplied iterate vector is not of the same length as complicating variables in subproblem!"
end
# fix complicating variables
@constraint(temp_mod2,fix[i=1:length(capa)],capa[i]==res[i])

# deduct investment costs from 

# solver settings (only Gurobi for now)
set_optimizer(temp_mod2,solver)
set_optimizer_attribute(temp_mod2,"Method",2)
set_optimizer_attribute(temp_mod2,"Crossover",0)
set_optimizer_attribute(temp_mod2,"BarConvTol",opt_gap)

# Optimize
@time optimize!(temp_mod2)
if termination_status(temp_mod2)!=OPTIMAL
    @error "Subproblem did not solve to optimality!"
    return temp_mod2
end
# Define output
ret = (obj = objective_value(temp_mod),duals=dual.(temp_mod[:fix]),res_obj=objective_value(temp_mod)-dual.(temp_mod[:fix])'*res);

temp_mod


t1 = copy(sub_mod_dict[2016])
t2 = copy(sub_mod_dict[2016])

set_optimizer(t1,Gurobi.Optimizer)
set_optimizer(t2,dual_optimizer(Gurobi.Optimizer))

@time optimize!(t1)
@time optimize!(t2)
