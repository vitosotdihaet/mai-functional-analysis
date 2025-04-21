from scipy.integrate import quad
import matplotlib.pyplot as plt
import numpy as np

def monomial_basis(t, n):
    return t**(n-1)

def y(t):
    return np.exp(-t)

def f(t):
    return 4*L/5 - t**2

def product(f1, f2) -> float:
    def product(t): return f1(t) * f2(t) * f(t)
    return quad(product, START, END)[0]

def gram_schmidt(basis_count: int) -> list:
    orthogonal_basis = []
    for i in range(1, basis_count + 1):
        def current_function(t, n=i): return monomial_basis(t, n)

        for prev_i in range(i - 1):
            previous_function = orthogonal_basis[prev_i]
            num = product(current_function, previous_function)
            denom = product(previous_function, previous_function)
            c = num / denom

            updated_function = (lambda f, c, p: lambda t: f(t) - c * p(t))(current_function, c, previous_function)
            current_function = updated_function

        orthogonal_basis.append(current_function)
    return orthogonal_basis

def fourier_coefficients(basis, term_count):
    c = []
    for b in basis[:term_count]:
        num = product(y, b)
        denom = product(b, b)
        c.append(num / denom)
    return c

def approximate_value(t, basis, coefficients):
    v = 0.0
    for c, b in zip(coefficients, basis):
        v += c * b(t)
    return v

def calculate_error(basis, coeffs):
    def error_func(t): return (y(t) - approximate_value(t, basis, coeffs))**2 * f(t)
    return np.sqrt(quad(error_func, START, END)[0])

def get_term_name(terms: int) -> str:
    name = 'член'
    if 1 < terms <= 4:
        name = 'члена'
    elif terms > 4:
        name = 'членов'
    return name


K = 3
L = 15

START = -0.8 - K/10
END = 0.8 + K/10

ts = np.linspace(START, END, 300)
original_values = y(ts)

accuracy = [1e-1, 1e-2, 1e-3]
approximation_results = {}

for eps in accuracy:
    terms = 1
    while True:
        b = gram_schmidt(terms)
        c = fourier_coefficients(b, terms)
        e = calculate_error(b, c)

        if e <= eps:
            approximation_results[eps] = (terms, e, b, c)
            break
        terms += 1


_, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 13))

ax1.plot(ts, original_values, 'k', lw=3, label='Исходная функция')
for eps in accuracy:
    terms, err, basis_funcs, c = approximation_results[eps]
    approximate_values = [approximate_value(t, basis_funcs, c) for t in ts]
    ax1.plot(ts, approximate_values, '--', label=f'{terms} {get_term_name(terms)}, ε={eps}')

ax1.set_xlabel('t', fontsize=12)
ax1.set_ylabel('Значение функции', fontsize=12)
ax1.set_title('Аппроксимации с разной точностью', pad=15)
ax1.legend()
ax1.grid(alpha=0.4)


term_count = max(approximation_results[eps][0] for eps in accuracy)
for terms in range(1, term_count + 1):
    b = gram_schmidt(terms)
    c = fourier_coefficients(b, terms)
    approximate_values = [approximate_value(t, b, c) for t in ts]
    ax2.plot(ts, approximate_values, '--', label=f'{terms} {get_term_name(terms)}')
ax2.plot(ts, original_values, 'k', lw=2, label='Исходная функция')

ax2.set_xlabel('t', fontsize=12)
ax2.set_ylabel('Значение функции', fontsize=12)
ax2.set_title('Аппроксимации функции с ростом числа членов базиса', pad=15)
ax2.legend()
ax2.grid(alpha=0.4)

plt.tight_layout()
plt.savefig('plot.png')
