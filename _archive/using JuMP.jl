using JuMP
using Gurobi
using BenchmarkTools

model = Model(Gurobi.Optimizer)
@variable(model, x >= 0)
@variable(model, 0 <= y <= 3)
@variable(model, z <= 1)
@objective(model, Min, 12x + 20y - z)
@constraint(model, c1, 6x + 8y >= 100)
@constraint(model, c2, 7x + 12y >= 120)
@constraint(model, c3, x + y <= 20)
optimize!(model)

@btime set_objective_coefficient(model,x,7)
@btime set_normalized_rhs(c1,110.0)
optimize!(model)
