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



function solve_subproblem(sub_m::JuMP.AbstractModel,res,opt_gap,capa_vars;solver=Gurobi.Optimizer)
    temp_mod = copy(sub_m) # copy model so as to not alter the default subproblem
    set_silent(temp_mod)
    # define complicating variables
    capa = filter(x -> (string(x) in capa_vars),all_variables(temp_mod))
    #capa = filter(x->contains(string.(x),"N_"),all_variables(temp_mod))
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
    optimize!(temp_mod)
    if termination_status(temp_mod)!=OPTIMAL
        @error "Subproblem did not solve to optimality!"
        return temp_mod
    end
    # Define output
    ret = (obj = objective_value(temp_mod),duals=dual.(temp_mod[:fix]),res_obj=objective_value(temp_mod)-dual.(temp_mod[:fix])'*res);
    return ret
end

#function solve_seq_subproblem(sub_m::JuMP.AbstractModel,capa_res,lvl_res,opt_gap,capa_vars,lds_lvl_vars,year;solver=Gurobi.Optimizer)
#    temp_mod = copy(sub_m)
#    set_silent(temp_mod)
#    capa = filter(x->(string(x) in capa_vars),all_variables(temp_mod))
#    lds_lvl = filter(x->(string(x) in lds_lvl_vars),all_variables(temp_mod))

    # fix capacities
#    @constraint(temp_mod,fix_capa[i=1:length(capa)],capa[i]==res[i])

    # fix storage levels
    
#    @constraint(temp_mod,fix_lvl[])



#end

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
    if settings.lds_mode
        @assert length(centre) == length(top_m[:compl])
    else
        @assert length(centre) == length(top_m[:capa])
    end
    set_optimizer(stab_top_m,settings.solver)
    set_silent(stab_top_m)
    l2_norm = settings.lds_mode ? @expression(stab_top_m,sum((stab_top_m[:compl][i] - centre[i])^2 for i in eachindex(centre))) : @expression(stab_top_m,sum((stab_top_m[:capa][i] - centre[i])^2 for i in eachindex(centre)))
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
            sub_dtr_dict[y] = create_sub_dict(deepcopy(sub_dtr),y)
            tmp = copy(sub_m)
            define_model!(tmp,sub_dtr_dict[y],maxhours=settings.maxhours)
            tmp[:capa] = filter(x->(string(x) in string.(keys(capas_dict))),all_variables(tmp))
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
            sub_dtr_dict[y] = create_sub_dict(deepcopy(sub_dtr),y)
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
    oracle_output = Dict{Int64,Dict{Union{Int64,String},NamedTuple}}()
    
    # initialize stability centre
    if settings.init_bool
        init_mod = Model(solver)
        define_model!(init_mod,sub_dtr_dict[settings.presolve_year],maxhours=settings.presolve_hours)
        optimize!(init_mod)
        x0 = value.(filter(x->(string(x) in string.(mstr[:compl])),all_variables(init_mod)))
        @assert length(x0) == length(mstr[:capa])
    else
        x0 = zeros(length(mstr[:capa]))
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

        # On-demand accuracy test
        approx_bool = false
        # compute target
        if !isempty(oracle_output) && oracle_count > settings.oda_orac_count
            target = settings.oda_kappa*lower_bound + (1-settings.oda_kappa)*upper_bound
            # compute scenario-wise lower approximation based on existing cuts
            approx_vec = []
            approx_idx = []
            for year in settings.year_set
                approx_tmp, idx = findmax([oracle_output[j][year].res_obj + oracle_output[j][year].duals'*res for j in sort(collect(keys(oracle_output)))])
                push!(approx_vec,approx_tmp)
                push!(approx_idx,sort(collect(keys(oracle_output)))[idx])
            end
            approx_f = sum(settings.year_prob[i]*approx_vec[i] for i in eachindex(approx_vec))
            if approx_f > target
                approx_bool = true 
            end
        end

        approx_count = approx_bool ? approx_count + 1 : 0
        #oracle_count = !approx_bool ? oracle_count + 1 : 0 
        #print(approx_bool)


        # Call oracle if approximation is below target
        if (!approx_bool) || approx_count > settings.oda_count 
            #@info "Calling oracle"
            tmp_dict = Dict{Union{String,Int64},NamedTuple}()
            con_exact_bool = []
            # This bit will be parallelised
            for year in settings.year_set
                tmp_dict[year] =  solve_subproblem(sub_mod_dict[year],res,compute_conv_tol(gap,settings.parameters.optimality_gap,settings.parameters.acc_tup);solver)
                # solve_oda_subproblem(sub_mod_dict[year],res,oracle_output,0.5*upper_bound+0.5*lower_bound,year,compute_conv_tol(gap,settings.parameters.optimality_gap,settings.parameters.acc_tup);settings.solver)
                #push!(con_exact_bool,exact_bool)
            end
            #exact_bool = con_exact_bool[1]
            # save output
            oracle_output[k] = tmp_dict
            approx_bool = false
            approx_count = 0 
        end

        oracle_count = !approx_bool ? oracle_count + 1 : 0 
            

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
            if !approx_bool
                for s in settings.year_set
                    cut = @constraint(mstr,
                                    mstr[:θ][s] >= oracle_output[k][s].obj - mstr[:investment_obj] + sum(oracle_output[k][s].duals[i]*(mstr[:capa][i] - res[i] ) for i in eachindex(mstr[:capa]))
                                );
                set_name(cut,"cut$(k)[$s]")
                end
            end
        else
            if approx_bool
                approx_idx = DenseAxisArray(approx_idx,settings.year_set)
                cut = @constraint(mstr,
                                mstr[:θ] >= sum(probs[s]*(oracle_output[approx_idx[s]][s].obj - mstr[:investment_obj] + oracle_output[approx_idx[s]][s].duals'*(mstr[:capa] - res)) for s in settings.year_set)
                            );    
            else
                cut = @constraint(mstr,
                                mstr[:θ] >= sum(probs[s]*(oracle_output[k][s].obj - mstr[:investment_obj] + sum(oracle_output[k][s].duals[i]*(mstr[:capa][i] - res[i] ) for i in eachindex(mstr[:capa]))) for s in settings.year_set)
                            );
            end
            set_name(cut,"cut$(k)")
        end

        # compute oracle objective value and expected descent
        v = upper_bound - lower_bound
        if !approx_bool
            f = sum(probs[s]*oracle_output[k][s].obj for s in settings.year_set)
        else
            f = upper_bound
        end
        #println(f)
        #f = !approx_bool ? sum(probs[s]*oracle_output[k][s].obj for s in settings.year_set) : f
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



function create_sub_dict(sub_dtr::DieterModel,year::Union{Int64,String})
    tmp_dtr = sub_dtr
    tmp_avail = CSV.read(joinpath(ts_path,"$(year)_availability.csv"),DataFrame)
    tmp_load = CSV.read(joinpath(ts_path,"$(year)_load.csv"),DataFrame)
    @assert unique(tmp_load[!,:h]) == unique(tmp_avail[!,:h])
    tmp_dtr.sets[:h] = unique(tmp_load[!,:h])
    if length(tmp_dtr.sets[:h])<8760
        for n in tmp_dtr.sets[:n], r in tmp_dtr.sets[:nondis], h in tmp_dtr.sets[:h]
            #if tmp_dtr.parameters[:gen_availability][n,r,h]
            tmp_dtr.parameters[:gen_availability][n,r,h] = filter(row->(row.n == "$(n)" && row.h == "$(h)"),tmp_avail)[:,Symbol("$(r)")][1]
            tmp_dtr.parameters[:node_electricity_demand][n,h] = filter(row->(row.n == "$(n)" && row.h == "$(h)"),tmp_load)[:,:load][1]
        end
        for n in tmp_dtr.sets[:n], sto in tmp_dtr.sets[:inflow], h in tmp_dtr.sets[:h]
            tmp_dtr.parameters[:sto_inflow][n,sto,h] = filter(row->(row.n == "$(n)" && row.h == "$(h)"),tmp_avail)[:,Symbol("$(sto)")][1]
        end
    else
        for n in tmp_dtr.sets[:n], r in tmp_dtr.sets[:nondis]
            tmp_dtr.parameters[:gen_availability][n,r,:] = filter(row->row.n == "$(n)",tmp_avail)[:,Symbol("$(r)")]
            tmp_dtr.parameters[:node_electricity_demand][n,:] = filter(row->row.n=="$(n)",tmp_load)[:,:load]
        end
        for n in tmp_dtr.sets[:n], sto in tmp_dtr.sets[:inflow]
            # import inflow data
            tmp_dtr.parameters[:sto_inflow][n,sto,:] = filter(row->row.n=="$(n)",tmp_avail)[:,Symbol("$(sto)")]
        end
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

function map_storage_compl(year_set::Vector{Int},top_m::JuMP.AbstractModel,sub_m::JuMP.AbstractModel)
    lds_lvl = top_m[:lds_lvl]
end