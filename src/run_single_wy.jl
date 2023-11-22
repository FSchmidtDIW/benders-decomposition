using DIETER
using JuMP
using Gurobi
using DataFrames
using CSV

b=pwd()
# Define paths
datapath = b*"/data/test_model/"
result_path = b*"/results/"
ts_path = b*"/data/ts/"


# define weather year
year = 1997#parse(Int,ARGS[1])

# Include functions
include("benders_functions.jl")

# Create dtr object
dtr = DieterModel(datapath; verbose=false);

# create dtr object with correct weather year
sub_dtr = create_sub_dict(dtr,year)


m = Model(Gurobi.Optimizer)
define_model!(m,sub_dtr,maxhours=8760)

JuMP.fix.(m[:G_INF],0,force=true)

set_optimizer_attribute(m,"NumericFocus",3)
set_optimizer_attribute(m,"Method",2)
set_optimizer_attribute(m,"Crossover",0)
optimize!(m)

if termination_status(m)!=OPTIMAL
    infeas = find_infeasible_constraints(m)
    infeas_df = DataFrame(:infeas => infeas)
    infeas_df |> CSV.write("$(year)_infeasibility_report.csv")
else
        # Select variables to export
    exog = [
        (:EnergyBalance, [:n, :h],          :Marginal, :Constraint),
        (:Z,             Array{Symbol,1}(), :Value,    :Objective),
        ]
    pars = collect_symbols(sub_dtr, m, :Parameter) # All parameters
    vars = collect_symbols(sub_dtr, m, :Variable)  # All variables

    symbs = [exog;pars;vars];
    collect_results(sub_dtr,m,symbs)

    for sym in keys(sub_dtr.results)
        sub_dtr.results[sym][:df] |> CSV.write(joinpath(result_path,"$(year)_$(sym)_output.csv"))
    end
end