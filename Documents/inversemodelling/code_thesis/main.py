from darts_interpolator import DartsInterpolator


# function for arbitrary number of dimensions
def func(my_vars, values):
    values[0] = sum([var * var for var in my_vars])
    values[1] = -sum([var * var for var in my_vars])
    return 0

n_dim = 2
axes_points = [7] * n_dim
axes_min = [-1] * n_dim
axes_max = [1.5] * n_dim


my_itor = DartsInterpolator(func, axes_points=axes_points, axes_min=axes_min, axes_max=axes_max)
result = my_itor.interpolate_point([1, 1.2])
print(result)

result, derivs = my_itor.interpolate_point_with_derivatives([1, 1.2])
print(result, derivs)

my_itor = DartsInterpolator(func, axes_points=axes_points, axes_min=axes_min, axes_max=axes_max, mode='adaptive')
result = my_itor.interpolate_point([1, 1.2])
print(result)

result, derivs = my_itor.interpolate_point_with_derivatives([1, 1.2])
print(result, derivs)
