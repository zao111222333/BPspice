from gspice import Expression, before_update
len = 200
iter = 10000
step = 0.01
x, x_ref = Expression.rand_uniform(len, -1, 1, True)
y, y_ref = Expression.rand_uniform(len, -1, 1, True)
f = x*x + y*y

print(f'Given random x,y, we want to reduce f=x^2+y^2')
print(f'Init\n x = {x}\n y = {y}')

for i in range(iter):
    f_value = f.value()
    if i%200 == 0:
        print(f'Iter {i} : f = {f_value}')
    grads = f.backward()
    df_dx = grads.take(x_ref)
    df_dy = grads.take(y_ref)
    before_update()
    x_ref.update(df_dx, lambda grad : -grad*step)
    y_ref.update(df_dy, lambda grad : -grad*step)

print(f'Iter {iter} : f = {f_value}')
print(f'Finally\n x = {x}\n y = {y}')