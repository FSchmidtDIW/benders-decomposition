#%% -----------------------------
# Reproducible example custom stopping rule for LP barrier based on dual bound
# Based on 

using JuMP
using Gurobi
using JSON


data = JSON.parse("""
{
    "plants": {
        "Seattle": {"capacity": 350},
        "San-Diego": {"capacity": 600}
    },
    "markets": {
        "New-York": {"demand": 300},
        "Chicago": {"demand": 300},
        "Topeka": {"demand": 300}
    },
    "distances": {
        "Seattle => New-York": 2.5,
        "Seattle => Chicago": 1.7,
        "Seattle => Topeka": 1.8,
        "San-Diego => New-York": 2.5,
        "San-Diego => Chicago": 1.8,
        "San-Diego => Topeka": 1.4
    }
}
""")
P = keys(data["plants"])
M = keys(data["markets"])
distance(p::String, m::String) = data["distances"]["$(p) => $(m)"]
model = direct_model(Gurobi.Optimizer())
@variable(model, x[P, M] >= 0)
@constraint(model, [p in P], sum(x[p, :]) <= data["plants"][p]["capacity"])
@constraint(model, [m in M], sum(x[:, m]) >= data["markets"][m]["demand"])
@objective(model, Min, sum(distance(p, m) * x[p, m] for p in P, m in M));

# enforcing Barrier 
set_optimizer_attribute(model,"Method",2)
# No cross over
set_optimizer_attribute(model,"Crossover",0)


# Define dual bound threshold
thr = 1e3

# Define callback
c = Float64[]
function absolute_stopping_criterion(cb_data,cb_where)
    objbound =Ref{Cdouble}()
    Gurobi.GRBcbget(cb_data,cb_where,Gurobi.GRB_CB_BARRIER_DUALOBJ,objbound)
    objbound = objbound[]
    push!(c,objbound)
    if maximum(c) > thr
        for p in P
            for m in M
                valuePdouble = Ref{Cdouble}()
                col = unsafe_backend(model).variable_info[x[p, m].index].column
                colc = Cint(col - 1)
                ret = GRBgetdblattrelement(unsafe_backend(model), "X", colc, valuePdouble)
                println("x = $(x[p,m]) has value $(valuePdouble[])")
            end
        end
        GRBterminate(backend(model).inner)
    end
    return nothing 
end

# Attach callback function to model
MOI.set(model,Gurobi.CallbackFunction(),absolute_stopping_criterion)


# Optimize model
optimize!(model)

# retrieve termination_status
termination_status(model) == MOI.INTERRUPTED

# Obtain function value and primal solution
objective_value(model)
value.(model[:x])

model