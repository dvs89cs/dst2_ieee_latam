###############################################################################
# El código corresponde al artículo "Iterative Numerical Method to Construct 
# Wavelets Matching Unidimensional Patterns" sometido a IEEE Latin American 
# Transactions (2022).
# Autor: Damian Valdés Santiago© (dvs89cs@gmail.com)
# Licencia MIT.
###############################################################################

from pickle import TRUE
from numpy.__config__ import show
from sympy import symbols, nonlinsolve, IndexedBase, Sum, Tuple, lambdify
import sympy
from random import random
from pprint import pprint
import numpy
import scipy.optimize
import matplotlib.pyplot as plt
from matplotlib import patches
from scipy.interpolate import interp1d
import pywt
from scipy import signal
from basic_units import radians
import pandas as pd
import random

def rescale(arr, factor=2):
    n = len(arr)
    return numpy.interp(numpy.linspace(0, n, factor*n+1), numpy.arange(n), arr)

def signaltonoise(a, axis=0, ddof=0):
    a = numpy.asanyarray(a)
    m = a.mean(axis)
    sd = a.std(axis=axis, ddof=ddof)
    return numpy.where(sd == 0, 0, m/sd)

def check_filter_linear_phase(filter, rtol=1e-05, atol=1e-08):
    """
    Return a logical expressing if the filter has linear phase.

    Parameters
    ----------
    filter : array
        Filter coefficients.
    rtol : float
        The relative tolerance parameter (see documentation for numpy.allclose).
    atol : float
        The absolute tolerance parameter (see documentation for numpy.allclose).

    Returns
    -------
    If the filter have linear phase.

    Notes
    -----
    A real-valued and causal FIR filter h have a (generalized) linear phase response if and only if 
    their coefficients satisfy either h[n] = h[N − 1 − n] (symmetry) or h[n] = −h[N−1−n] (anti-symmetry) 
    where N is the filter length (number of taps).
    """
    # Symmetry or anti-symmetry
    filter_module = numpy.abs(filter)
    return numpy.linalg.norm(filter_module - numpy.flip(filter_module), ord=2) or numpy.linalg.norm(filter_module + numpy.flip(filter_module), ord=2)

def measure_s(data, alpha=0.15):
    """
    Compute the normalized similarity measure S from Guido (2018) that 
    emphasizes the presence of zeros in the data turning them into the unity.
    
    Return an array corresponding to the measure S of data.

    Parameters
    ----------
    data : array
        Data over the measure S is computed.
    alpha : float
        Parameter of the measure S.

    Returns
    -------
    The measure S of the data.

    Notes
    -----
    The measure S definition is from Guido, R. C. (2018). Fusing time, frequency 
    and shape-related information: Introduction to the Discrete Shapelet Transform’s 
    second generation (DST-II). Information Fusion, 41, 9–15. 
    https://doi.org/10.1016/j.inffus.2017.07.004

    The measure S is e^{-|data|^alpha}. S tends to a constant k near 0, and tends to 0 for the rest of values.
    If alpha tends to infinity, then the constant k tends to infinity.
    """
    return numpy.exp(-numpy.abs(data)**alpha)

def plot_coeffs(data, w, level, title):
    """Show dwt coefficients for given data and wavelet."""
    a = data
    ca = []
    cd = []

    for i in range(level):
        (a, d) = pywt.dwt(a, w)
        ca.append(a)
        cd.append(d)

    fig = plt.figure()
    ax_main = fig.add_subplot(len(ca) + 1, 1, 1)
    ax_main.set_title(title)
    ax_main.plot(data)
    ax_main.set_xlim(0, len(data) - 1)

    for i, x in enumerate(ca):
        ax = fig.add_subplot(len(ca) + 1, 2, 3 + i * 2)
        ax.plot(x, 'r')
        ax.set_ylabel("A%d" % (i + 1))
        ax.set_xlim(0, len(x) - 1)

    for i, x in enumerate(cd):
        ax = fig.add_subplot(len(cd) + 1, 2, 4 + i * 2)
        ax.plot(x, 'g')
        ax.set_ylabel("D%d" % (i + 1))
        # Scale axes
        ax.set_xlim(0, len(x) - 1)
        ax.set_ylim(min(0, 1.4 * min(x)), max(0, 1.4 * max(x)))

    plt.show()

def dst2(pattern, initial_approximation_type='null-vector'):
    """
    Find a wavelet adapted to the pattern (shapelet) using the algorithm 
    Discrete Shapelet Transform II (DST-II) from Guido (2017).

    Return the wavelet object from PyWavelets adapted to the pattern.

    Parameters
    ----------
    pattern : array
        One dimensional array containing the points of the pattern.
    initial_approximation_type: str
        Type of initial approximation. Should be one of
            - 'null-vector'      : null vector of size N-1, where N is the length of the pattern
            - 'lm'               : Solves the system of nonlinear equations in a least squares
                                   sense using a modification of the Levenberg-Marquardt algorithm as
                                   implemented in MINPACK.
            - 'broyden1'         : Uses Broyden's first Jacobian approximation, it is
                                   known as Broyden's good method.
            - 'broyden2'         : Uses Broyden's second Jacobian approximation, it
                                   is known as Broyden's bad method.
            - 'anderson'         : Uses (extended) Anderson mixing.
            - 'linearmixing'     : Uses a scalar Jacobian approximation.
            - 'excitingmixing'   : Uses a tuned diagonal Jacobian.
            - 'krylov'           : Uses Krylov approximation for inverse Jacobian. It
                                   is suitable for large-scale problem.
            - 'df-sane'          : Derivative-free spectral method.
    Returns
    -------
    shapelet : wavelet objetc from PyWavelet
        The wavelet adapted to the pattern.
    phi_d : array
        An array with the scaling function for decomposition.
    psi_d : array
        An array with the wavelet function for decomposition.
    phi_r : array
        An array with the scaling function for reconstruction.
    psi_r : array
        An array with the wavelet function for reconstruction.
    x : array
        x-points for the evaluation of the scaling and decomposition functions.
    nfev : integer
        Number of function calls during the iteration to solve the related non-linear equation system.
    fvec_norm2 : integer
        Norm 2 of the function evaluated at the output (fvec) resulting for the iteration to solve 
        the related non-linear equation system.
    mesg : str
        Convergence message about the iteration to solve the related non-linear equation system.
    Notes
    -----
    This method follows the algorithm for constructing a wavelet adapted to a given pattern from 
    Guido, R. C. (2018). Fusing time, frequency and shape-related information: 
    Introduction to the Discrete Shapelet Transform’s second generation (DST-II). 
    Information Fusion, 41, 9–15. https://doi.org/10.1016/j.inffus.2017.07.004

    This algorithm consists in solving a non-linear equation system formed by
    unitary energy, vanishing moments, orthogonality and matching conditions.

    """
    # Filter size
    N = len(pattern) - 1

    # Index variables for symbolic expressions
    k = symbols('k', integer=True)
    b = symbols('b', integer=True)
    l = symbols('l', integer=True)
    q = symbols('q', cls=IndexedBase)
    m = symbols('m', cls=IndexedBase)

    # Symbolic expression for unitary energy condition
    unitary_energy_condition = Sum(q[k]**2, (k, 0, N-1)).doit() - 1
    unitary_energy_condition = unitary_energy_condition.subs([(q[i], "q{0}".format(i)) for i in range(N)])

    # Symbolic expression for vanishing moments condition
    vanishing_moments_equations = []

    for b in range(0, N//2-2):
        temporal_equation = Sum(q[k] * (k)**b, (k, 0, N-1)).doit()
        temporal_equation = temporal_equation.subs([(q[i], "q{0}".format(i)) for i in range(N)])
        vanishing_moments_equations.append(temporal_equation)

    # Symbolic expression for orthogonality condition
    orthogonality_equations = []

    for l in range(1, N//2):
        temporal_equation = Sum(q[k] * q[k + 2*l], (k, 0, N-2*l-1)).doit()
        temporal_equation = temporal_equation.subs([(q[i], "q{0}".format(i)) for i in range(N)])
        orthogonality_equations.append(temporal_equation)

    # Symbolic expressions for matching conditions
    matching_condition_1 = Sum(q[k] * m[k + 1], (k, 0, N-1)).doit()
    matching_condition_1 = matching_condition_1.subs([(m[i], pattern[i]) for i in range(len(pattern))])
    matching_condition_1 = matching_condition_1.subs([(q[i], "q{0}".format(i)) for i in range(N)])

    matching_condition_2 = Sum(q[k] * m[k], (k, 0, N-1)).doit()
    matching_condition_2 = matching_condition_2.subs([(m[i], pattern[i]) for i in range(len(pattern))])
    matching_condition_2 = matching_condition_2.subs([(q[i], "q{0}".format(i)) for i in range(N)])

    # Non-linear equations system including all the previous equations
    all_equations = []
    all_equations.append(unitary_energy_condition)
    [all_equations.append(eq) for eq in vanishing_moments_equations]
    [all_equations.append(eq) for eq in orthogonality_equations]
    all_equations.append(matching_condition_1)
    all_equations.append(matching_condition_2)

    # Numerically solving the previous non-linear equations system 
    v = symbols('q0:%d'%N)

    # Solving the non-linear equations system with the modified Powell method 
    # (see scipy.optimize.fsolve documentation for details)
    f = lambdify([v], all_equations, 'scipy')

    null_vector = [0 for i in v]

    # Null vector for default initial approximation
    initial_approximation = null_vector

    # Selecting the initial approximation to solve by numerical iterations the non-linear system f
    if initial_approximation_type == 'lm':
        # Method lm solves the system of nonlinear equations in a least squares sense using a modification of
        # the Levenberg-Marquardt algorithm as implemented in MINPACK 
        initial_approximation = scipy.optimize.root(f, null_vector, method='lm').x
    elif initial_approximation_type == 'df-sane':
        # Method df-sane is a derivative-free spectral method. 
        initial_approximation = scipy.optimize.root(f, null_vector, method='df-sane').x
    elif initial_approximation_type == 'broyden1':
        # Method broyden1 uses Broyden’s first Jacobian approximation, it is known as Broyden’s good method.
        initial_approximation = scipy.optimize.root(f, null_vector, method='broyden1').x
    elif initial_approximation_type == 'broyden2':
        # Method broyden2 uses Broyden’s second Jacobian approximation, it is known as Broyden’s bad method.
        initial_approximation = scipy.optimize.root(f, null_vector, method='broyden2').x
    elif initial_approximation_type == 'anderson':
        # Method anderson uses (extended) Anderson mixing.
        initial_approximation = scipy.optimize.root(f, null_vector, method='anderson').x
    elif initial_approximation_type == 'Krylov':
        # Method Krylov uses Krylov approximation for inverse Jacobian. It is suitable for large-scale problem.
        initial_approximation = scipy.optimize.root(f, null_vector, method='Krylov').x
    elif initial_approximation_type == 'linearmixing':
        # Method linearmixing uses a scalar Jacobian approximation.
        initial_approximation = scipy.optimize.root(f, null_vector, method='linearmixing').x
    elif initial_approximation_type == 'excitingmixing':
        # Method excitingmixing
        initial_approximation = scipy.optimize.root(f, null_vector, method='excitingmixing').x

    # Getting the solution vector q (corresponding to the high frequency descomposition filter)
    # using the modified Powell hybrid method (HYBRD in MINIPACK): 
    # Two of its main characteristics involve the choice of the correction as a convex combination 
    # of the Newton and scaled gradient directions, and the updating of the Jacobian by the 
    # rank-1 method of Broyden. The choice of the correction guarantees (under reasonable conditions) 
    # global convergence for starting points far from the solution and a fast rate of convergence. 
    # The Jacobian is approximated by forward differences at the starting point, but forward differences
    # are not used again until the rank-1 method fails to produce satisfactory progress.
    # This method returns the solution q, the dict info with numerical information, and
    # an integer flag (ier) that it is set to 1 if a solution was found.
    q, info, ier, mesg = scipy.optimize.fsolve(f, initial_approximation, full_output=1)

    # Number of function calls (nfev)
    nfev = info['nfev']

    # The norm 2 of the function evaluated at the output (fvec)
    fvec_norm2 = numpy.linalg.norm(info['fvec'], ord=2) 

    # Getting the vector p (corresponding to the low frequency decomposition filter)
    p = [(-1)**(k + 1) * q[N - k - 1] for k in range(len(q))]

    # Getting the vector p_hat (corresponding to the low frequency reconstruction filter)
    p_hat = [p[N - k - 1] for k in range(len(p))]
    
    # Getting the solution vector q_hat (corresponding to the high frequency reconstruction filter)
    q_hat = [(-1)**(k + 1) *q[k] for k in range(len(q))]

    # Creating the filter bank from the previous filters
    filter_bank = [p, q, p_hat, q_hat]

    # Constructing the shapelet from the filter bank with PyWavelets package
    shapelet = pywt.Wavelet(name="Shapelet", filter_bank=filter_bank)

    # Creating the minor shapelet (phi) and the major shapelet (psi) for decomposition and
    # reconstruction using the cascade algorithm of PyWavelets with 8 levels.
    # This allows to plot the scaling and wavelet functions corresponding to the shapelet.
    # The returned x refers to the x-points for plotting the functions.
    phi_d, psi_d, phi_r, psi_r, x = shapelet.wavefun(level=8)

    return shapelet, phi_d, psi_d, phi_r, psi_r, x, nfev, fvec_norm2, mesg

def plot_adapted_wavelet(pattern, shapelet, x, phi_d, psi_d, save_to=None):
    # Frequency response of filters from DST-II in pi
    w_q, h_q = signal.freqz(shapelet.dec_hi)
    w_p, h_p = signal.freqz(shapelet.dec_lo)

    # # Frequency response of filters from DST-II in 2*pi
    # w_q, h_q = signal.freqz(shapelet.dec_lo, whole=True)
    # w_p, h_p = signal.freqz(shapelet.dec_hi, whole=True)

    # Plotting the pattern to detect, the obtained filters, the minor and 
    # major shapelets, and the frequency response of the filters 
    points = range(0, len(pattern))
    f_cubic = interp1d(points, pattern, kind='cubic')

    fig = plt.figure(constrained_layout=True)
    gs = fig.add_gridspec(2, 4)
    fig_ax1 = fig.add_subplot(gs[0, 0])
    fig_ax1.set_title(r'Patrón $m$ a detectar')
    fig_ax1.stem(points, pattern, use_line_collection=True)
    fig_ax1.plot(points, f_cubic(points), '-', label='patrón')

    fig_ax2 = fig.add_subplot(gs[0, 1])
    fig_ax2.set_title(r'Filtro $q$')
    fig_ax2.stem(range(0, len(shapelet.dec_hi)), shapelet.dec_hi, use_line_collection=True)

    fig_ax9 = fig.add_subplot(gs[1, 0])
    fig_ax9.set_title(r'Diagrama de ceros')
    unit_circle = patches.Circle((0,0), radius=1, fill=False, linewidth=1.5, color='k', ls='dashed')
    fig_ax9.add_patch(unit_circle)
    z, p, k = signal.tf2zpk(shapelet.dec_hi, a=1)
    fig_ax9.scatter(numpy.real(p), numpy.imag(p),marker='o')
    l31 = fig_ax9.scatter(numpy.real(z), numpy.imag(z),marker='o', label='descomposición')
    z, p, k = signal.tf2zpk(shapelet.rec_hi, a=1)
    fig_ax9.scatter(numpy.real(p), numpy.imag(p), marker='x')
    l32 = fig_ax9.scatter(numpy.real(z), numpy.imag(z), marker='x', label='reconstrucción')
    fig_ax9.axvline(0, color='0.7')
    fig_ax9.axhline(0, color='0.7')
    fig_ax9.set_aspect('equal', adjustable="datalim")
    fig_ax9.legend(handles=[l31, l32], loc='lower left', bbox_to_anchor= (-0.2, -0.2), ncol=2,
                    borderaxespad=0)

    fig_ax3 = fig.add_subplot(gs[1, 1])
    fig_ax3.set_title(r'Filtro $p$')
    fig_ax3.stem(range(0, len(shapelet.dec_lo)), shapelet.dec_lo, use_line_collection=True)

    fig_ax4 = fig.add_subplot(gs[0, 2])
    fig_ax4.set_title('Shapelet menor')
    fig_ax4.plot(x, psi_d, 'tab:orange')

    fig_ax5 = fig.add_subplot(gs[1, 2])
    fig_ax5.set_title('Shapelet mayor')
    fig_ax5.plot(x, phi_d, 'tab:orange')

    fig_ax6 = fig.add_subplot(gs[0, 3])
    fig_ax6.set_title(r'Respuesta frecuencia/fase de $q$')
    fig_ax6.plot(w_q, abs(h_q), 'y')
    fig_ax6.set_ylabel('Magnitud', color='y')
    fig_ax6.set_xlabel('Frequencia [rads]')
    fig_ax6.set_xticks([0, numpy.pi/2, numpy.pi])
    fig_ax6.set_xticklabels([r'$0$', r'$\frac{\pi}{2}$', r'$\pi$'])

    ax2 = fig_ax6.twinx()
    angles = numpy.unwrap(numpy.angle(h_q))
    ax2.plot(w_q, angles, 'g')
    ax2.set_ylabel('Fase ([rads]', color='g')
    fig_ax6.grid()

    fig_ax7 = fig.add_subplot(gs[1, 3])
    fig_ax7.set_title(r'Respuesta frecuencia/fase de $p$')
    fig_ax7.plot(w_p, abs(h_p), 'y')
    fig_ax7.set_ylabel('Magnitud', color='y')
    fig_ax7.set_xlabel('Frequencia [rads]')
    fig_ax7.set_xticks([0, numpy.pi/2, numpy.pi])
    fig_ax7.set_xticklabels([r'$0$', r'$\frac{\pi}{2}$', r'$\pi$'])

    ax3 = fig_ax7.twinx()
    angles = numpy.unwrap(numpy.angle(h_p))
    ax3.plot(w_p, angles, 'g')
    ax3.set_ylabel('Fase [rads]', color='g')
    fig_ax7.grid()

    # Maximize the figure for better visualization
    mng = plt.get_current_fig_manager()
    mng.resize(*mng.window.maxsize())

    plt.show()

    if save_to is not None:
        fig.savefig(save_to, dpi=300)

def plot_pattern_detection(signal, adapted_wavelet, pattern, pattern_localisation, save_to=None):
    # One-level Discrete Shapelet Transform (DST-II_1) using the generated PyWavelet object
    (cA, cD) = pywt.dwt(signal, wavelet=adapted_wavelet, mode='per')

    dst2 = numpy.concatenate((cA, cD), axis=0)

    # Position of the pattern estimated in the index of the measure S maximum
    k = numpy.argmax(measure_s(cD))
    # Decomposition level used in DST-II
    j = 1
    # The pattern start at localisation_1: in Guido (2008) is k*2**j - 1, but he has 1-index
    localisation_1 = k*2**j
    # or at localisation_2: in Guido (2008) is k*2**j, but he has 1-index
    localisation_2 = k*2**j + 1

    # Plotting the values of measure S for the signal with the pattern embedded after DST-II
    fig, axs = plt.subplots(3, 1, sharex=True)

    axs[0].set_title('Señal con patrón')
    axs[0].plot(numpy.arange(len(signal)), signal)
    x_pattern = numpy.arange(pattern_localisation, pattern_localisation + len(pattern))
    axs[0].scatter(x_pattern, signal[x_pattern], color='g') 
    axs[0].axhline(y=0, color='r')
    l4 = axs[0].axvline(x=localisation_1, color='y', label='Predicción de la ocurrencia del patrón')
    axs[0].axvline(x=localisation_2, color='y')
    a=numpy.arange(len(signal))
    axs[0].set_xticks(a)
    axs[0].set_xticklabels(a, fontsize = 7)
    axs[0].margins(x=0)
    axs[0].xaxis.set_tick_params(labelbottom=True)

    axs[1].set_title('cA + cD de la señal con patrón')
    axs[1].axhline(y=0, color='r')
    l1 = axs[1].axvline(x=len(cA), color='purple', ls='--', label='Separación entre cA y cD')
    l2 = axs[1].axvline(x=k + len(cA), color='teal', ls='-.', label=r'Deteccion del patrón en cD según máximo de medida $\mathbb{S}$')
    axs[1].plot(dst2)
    axs[1].set_xticks(a)
    axs[1].set_xticklabels(a, fontsize = 7)
    axs[1].margins(x=0)
    axs[1].xaxis.set_tick_params(labelbottom=True)

    axs[2].set_title(r'Medida $\mathbb{S}$ de cD')
    axs[2].axvline(x=len(cA), color='purple', ls='--')
    axs[2].axvline(x=k + len(cA), color='teal', ls='-.')
    s_measure = measure_s(dst2, alpha=0.1)
    axs[2].plot(a[len(cA):len(a)], s_measure[len(cA):len(a)])
    # axs[2].plot(s_measure[0:len(cA)-1], color='white')
    axs[2].set_xticks(a)
    axs[2].set_xticklabels(a, fontsize = 7)
    axs[2].margins(x=0)
    axs[2].legend(handles=[l1, l2, l4], ncol=4, loc='lower left', bbox_to_anchor=(0.02, -0.5), prop={"size":10})
    plt.subplots_adjust(top=None, hspace=0.5)

    # Maximize the figure for better visualization
    mng = plt.get_current_fig_manager()
    mng.resize(*mng.window.maxsize())

    plt.show()

    if save_to is not None:
        fig.savefig(save_to, dpi=300)
    
    return localisation_1

def plot_measure_s_various_filters(signal, pattern_ocurrence, coeff_position, adapted_wavelet):
    a=numpy.arange(len(signal))
    
    list_of_wavelets = ['haar', 'db4', 'db6', 'db8', 'db10', 'db20', 'db30', 'db38', 'sym8', 'sym16', 'coif6', 'coif12', 'coif16']
    
    measure_s_maxs = []
    
    for base in list_of_wavelets:
        wavelet_base = pywt.Wavelet(base)
        (cA, cD) = pywt.dwt(signal, wavelet=wavelet_base, mode='per')
        s_measure = measure_s(numpy.concatenate((cA, cD), axis=0), alpha=0.1)
        measure_s_maxs.append(numpy.argmax(measure_s(cD)))
        plt.plot(a[len(cA):len(a)], s_measure[len(cA):len(a)], label=base)

    # One-level Discrete Shapelet Transform (DST-II_1) using the generated PyWavelet object
    (cA, cD) = pywt.dwt(signal, wavelet=adapted_wavelet, mode='per')

    dst2 = numpy.concatenate((cA, cD), axis=0)

    s_measure = measure_s(dst2, alpha=0.1)
    
    measure_s_maxs.append(numpy.argmax(measure_s(cD)))
    
    plt.plot(a[len(cA):len(a)], s_measure[len(cA):len(a)], color='k', linewidth=3, label='shapelet')
    plt.xticks(a[len(cA):len(a)], fontsize = 10)
    plt.margins(x=0)
    plt.axvline(x=coeff_position, color='k', ls='--')
    plt.legend(loc="upper right", ncol=3, shadow=True, fancybox=True)
    plt.title(r'Medida $\mathbb{S}$ de los coeficientes de detalles para varios filtros wavelet y la shapelet estimada')
    plt.ylabel(r'Valor de la medida $\mathbb{S}$')
    plt.xlabel('Posición del coeficiente de detalle')
    plt.show()
    
    prediction_positions = numpy.array(measure_s_maxs)*2 + 1
    prediction_errors = numpy.abs(pattern_ocurrence * numpy.ones_like(prediction_positions) - numpy.array(prediction_positions))
    
    list_of_wavelets.append('shapelet')
    statistics_detection = pd.DataFrame({'Initial approximation with': list_of_wavelets,
                              'Predicted position by measure S': prediction_positions,
                              'Prediction error': prediction_errors})

    statistics_detection.to_excel(path + 'patron_statistics_comparison.xlsx', sheet_name='Sheet1', 
                        columns=['Initial approximation with', 'Predicted position by measure S', 'Prediction error'])

    # statistics_detection.to_excel(path + 'patron2021_statistics_comparison.xlsx', sheet_name='Sheet1', 
    #                     columns=['Initial approximation with', 'Predicted position by measure S', 'Prediction error'])


# ############################################
# # STUDY ON MEASURE S
# ############################################

# alphas = [0.15, 0.2, 0.4, 0.8, 1]

# x = numpy.linspace(-4, 4, 1000)

# for i in alphas:
#     plt.plot(x, measure_s(x, alpha=i), label=r'$\alpha = ${0}'.format(i))
    
# plt.legend()  
# plt.title(r'Comportamiento de la medida $\mathbb{S}$ para diferentes valores de $\alpha$')
# plt.show()

# ############################################


############################################
# EXPERIMENTS WITH DST-II
############################################

#-------------------------------------------
# Adapting a wavelet for the same pattern in Guido (2018)
#-------------------------------------------

# # Pattern to detect in Guido (2018)
# pattern = [0.20, 0.50, 0.45, 0.85, 0.80, -0.75, 0.25, 0.20, 0.55]

# # Pattern in Guido (2021)
# pattern = [0.01, 0.01, 0.02, 0.01, 0.05, 0.23, 0.62, 0.90, 0.98, 0.88, 0.02, 0.01, 0.01]

# # initial_guesses = ['null-vector', 'lm', 'broyden1', 'broyden2', 'anderson', 
# #                     'linearmixing', 'excitingmixing', 'krylov', 'df-sane']

# initial_guesses = ['null-vector', 'linearmixing', 'anderson']

# # path = 'D:/Research/Mi PhD/Articulo 1 RIO 2022/Experimento 1/Patron 2018/'
# path = 'D:/Research/Mi PhD/Articulo 1 RIO 2022/Experimento 1/Patron 2021/'

# guess_names = []
# high_pass_filters = []
# low_pass_filters = []
# high_pass_reconstruction_filters = []
# low_pass_reconstruction_filters = []
# nfevs = []
# fvec_norms2 = []
# mesgs = []
# distance_to_guido_solution = []
# # guido_dst2_q = numpy.array([-0.0834, 0.1505, 0.5719, -0.7055, -0.0091, -0.2784, 0.2277, 0.1263])
# guido_dst3_q = numpy.array([-0.0069890396, -0.0047282445, -0.0162466459, 0.0210266730, 0.0760407388, 0.2581313042, 
#                            -0.8041341124, 0.5207860961, 0.0419502235, -0.0847506178, 0.0022720543, -0.0033584299])
# detection_predictions = []

# # # Signal with the pattern from Guido (2018) embedded
# # s1 = [numpy.cos(27/8 * numpy.pi * i) * numpy.sin(75/8 * numpy.pi * i) for i in range(0, 41)]
# # s2 = [numpy.cos(295/32 * numpy.pi * i) * numpy.sin(105/32 * numpy.pi * i) for i in range(50, 64)]
# # s = numpy.concatenate((s1, pattern, s2), axis=0)

# pattern_ocurrence = 41

# # Signal with the pattern from Guido (2021) embedded
# s1 = [numpy.cos(3.5 * numpy.pi * i) * numpy.sin(31.25 * numpy.pi * i) for i in range(0, 41)]
# s2 = [numpy.cos(13.281 * numpy.pi * i) * numpy.sin(3.906 * numpy.pi * i) for i in range(54, 64)]
# s = numpy.concatenate((s1, pattern, s2), axis=0)

# # w_modes = ['periodization', 'symmetric', 'reflect', 'periodic', 'antisymmetric','antireflect', 
# #             'smooth', 'constant', 'zero']

# for guess in initial_guesses:
#     # Constructing the shapelet adapted to the pattern
#     shapelet, phi_d, psi_d, phi_r, psi_r, x, nfev, fvec_norm2, mesg = dst2(pattern, initial_approximation_type=guess)

#     guess_names.append(guess)
#     high_pass_filters.append(shapelet.dec_hi)
#     low_pass_filters.append(shapelet.dec_lo)
#     nfevs.append(nfev)
#     fvec_norms2.append(fvec_norm2)
#     mesgs.append(mesg)
#     high_pass_reconstruction_filters.append(shapelet.rec_hi)
#     low_pass_reconstruction_filters.append(shapelet.rec_lo)
#     # distance_to_guido_solution.append(round(numpy.linalg.norm(numpy.abs(guido_dst2_q - shapelet.dec_hi), ord=2), 2))
#     distance_to_guido_solution.append(round(numpy.linalg.norm(numpy.abs(guido_dst3_q - shapelet.dec_hi), ord=2), 2))

#     # file_name = path + 'pattern2018_' + guess + '_' + 'powell.png'
#     file_name = path + 'pattern2021_' + guess + '_' + 'powell.png'

#     plot_adapted_wavelet(pattern, shapelet, x, phi_d, psi_d, save_to=file_name)

#     # file_name_detection = path + 'pattern2018_detection_' + guess + '_' + 'powell.png'  
#     file_name_detection = path + 'pattern2021_detection_' + guess + '_' + 'powell.png'

#     localisation = plot_pattern_detection(s, shapelet, pattern, pattern_localisation=41, save_to=file_name_detection)
#     detection_predictions.append(localisation)


# prediction_errors = numpy.abs(pattern_ocurrence * numpy.ones_like(detection_predictions) - numpy.array(detection_predictions))

# statistics_db = pd.DataFrame({'Initial approximation with': guess_names,
#                               'Filter q': high_pass_filters,
#                               'Function evals': nfevs,
#                               'Solution Norm2': fvec_norms2,
#                               'Convergence': mesgs,
#                               'Distance to Guido solution': distance_to_guido_solution,
#                               'Filter p': low_pass_filters,
#                               'Reconstruccion filter q': high_pass_reconstruction_filters,
#                               'Reconstruccion filter p': low_pass_reconstruction_filters})

# # statistics_db.to_excel(path + '/patron2018_statistics.xlsx', sheet_name='Sheet1', 
# #                     columns=['Initial approximation with', 'Function evals', 'Solution Norm2', 'Convergence', 
# #                             'Distance to Guido solution', 'Filter q', 'Filter p', 'Reconstruccion filter q', 'Reconstruccion filter p'])

# statistics_db.to_excel(path + '/patron2021_statistics.xlsx', sheet_name='Sheet1', 
#                     columns=['Initial approximation with', 'Function evals', 'Solution Norm2', 'Convergence', 
#                             'Distance to Guido solution', 'Filter q', 'Filter p', 'Reconstruccion filter q', 'Reconstruccion filter p'])


# statistics_detection = pd.DataFrame({'Initial approximation with': guess_names,
#                               'Predicted position by measure S': detection_predictions,
#                               'Prediction error': prediction_errors})

# # statistics_detection.to_excel(path + '/patron2018_statistics_detection.xlsx', sheet_name='Sheet1', 
# #                     columns=['Initial approximation with', 'Predicted position by measure S', 'Prediction error'])

# statistics_detection.to_excel(path + '/patron2021_statistics_detection.xlsx', sheet_name='Sheet1', 
#                     columns=['Initial approximation with', 'Predicted position by measure S', 'Prediction error'])

# #-------------------------------------------

#-------------------------------------------
# Comparing shapelet with other wavelet filters
#-------------------------------------------

# # Pattern to detect in Guido (2018)
# pattern = [0.20, 0.50, 0.45, 0.85, 0.80, -0.75, 0.25, 0.20, 0.55]

# Pattern in Guido (2021)
pattern = [0.01, 0.01, 0.02, 0.01, 0.05, 0.23, 0.62, 0.90, 0.98, 0.88, 0.02, 0.01, 0.01]

initial_guesses = ['anderson']

# path = 'D:/Research/Mi PhD/Articulo 1 RIO 2022/Experimentos/Patron 2018/'
path = 'D:/Research/Mi PhD/Articulo 1 RIO 2022/Experimentos/Patron 2021/'

# # Signal with the pattern from Guido (2018) embedded
# s1 = [numpy.cos(27/8 * numpy.pi * i) * numpy.sin(75/8 * numpy.pi * i) for i in range(0, 41)]
# s2 = [numpy.cos(295/32 * numpy.pi * i) * numpy.sin(105/32 * numpy.pi * i) for i in range(50, 64)]
# s = numpy.concatenate((s1, pattern, s2), axis=0)

# Signal with the pattern from Guido (2021) embedded
s1 = [numpy.cos(3.5 * numpy.pi * i) * numpy.sin(31.25 * numpy.pi * i) for i in range(0, 41)]
s2 = [numpy.cos(13.281 * numpy.pi * i) * numpy.sin(3.906 * numpy.pi * i) for i in range(54, 64)]
s = numpy.concatenate((s1, pattern, s2), axis=0)

# Constructing the shapelet adapted to the pattern
shapelet, phi_d, psi_d, phi_r, psi_r, x, nfev, fvec_norm2, mesg = dst2(pattern, initial_approximation_type='anderson')

pattern_position = 41

coeff_position = 52

plot_measure_s_various_filters(s, pattern_position, coeff_position, shapelet)

#-------------------------------------------

############################################
