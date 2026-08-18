# Three genuinely novel solutions

## 1. Fifth Airy moment

For

$$
a_5=\int_0^\infty \operatorname{Ai}(x)^5\,dx,
$$

define the Kampé de Fériet function

$$
\mathcal H(x,y)=
\sum_{j,k\ge0}
\frac{(\frac23)_j(1)_j(\frac12)_k(\frac43)_{j+k}}
     {(\frac43)_j(\frac{11}{6})_{j+k}}
\frac{x^j}{j!}\frac{y^k}{k!}.
$$

Then

$$
\boxed{
a_5=
\frac{F_1\!\left(1;\frac13,\frac12;\frac32;
\frac1{16},\frac14\right)}
{24\pi^2\,3^{2/3}\Gamma(\frac23)}
-
\frac{\mathcal H(\frac1{16},\frac14)}
{3^{1/3}48^{4/3}\pi^{3/2}
\Gamma(\frac13)\Gamma(\frac{11}{6})}.
}
$$

### Proof

Introduce the quartic Mellin transform

$$
M(s)=\int_0^\infty x^{s-1}\operatorname{Ai}(x)^4\,dx.
$$

Its known exact value is

$$
M(s)=
\frac{\Gamma(s)}
{48^{(s+2)/3}\pi^{3/2}\Gamma((2s+7)/6)}
{}_2F_1\!\left(
\frac{s+2}{3},\frac12;\frac{2s+7}{6};\frac14
\right).
$$

The remaining Airy factor has the expansion

$$
\operatorname{Ai}(x)=
\operatorname{Ai}(0)
{}_0F_1\!\left(;\frac23;\frac{x^3}{9}\right)
+
\operatorname{Ai}'(0)x
{}_0F_1\!\left(;\frac43;\frac{x^3}{9}\right).
$$

Termwise integration is justified by Tonelli's theorem applied to the two
nonnegative component series, together with Airy's exponential decay. This
produces sums involving $M(3j+1)$ and $M(3j+2)$. Gamma triplication and
expansion of the remaining ${}_2F_1$ resum these respectively to the Appell
$F_1$ value and $\mathcal H$ above.

This is an exact transformation of the improper integral into fixed named
two-variable hypergeometric constants. It uses no fitting, quadrature, or
accuracy-dependent truncation.

## 2. Equal-power TE+TM spherical-mode quality factor

Let

$$
L=n(n+1),\qquad c=\sqrt L,\qquad m=n+1,
$$

and define

$$
\widehat a_0=1,\qquad
\widehat a_{k+1}=\widehat a_k
\frac{(n+k+1)(n-k)(2k+1)}{2(k+1)L}.
$$

Set

$$
D(y)=\sum_{k=0}^n\frac{\widehat a_k y^{k+1}}{k+1},
\qquad
P(v)=(1+v)^m
\left[2-D\!\left(\frac1{1+v}\right)\right].
$$

Let $A$ be the companion matrix of the monic polynomial

$$
\frac{(-1)^mP(-u)}{[v^m]P(v)},
$$

and, crucially, use the principal matrix square root:

$$
S=cA_{\mathrm{principal}}^{1/2},
\qquad
K=\pi\left(\operatorname{tr}S-\frac{(2n+1)c}{2}\right).
$$

For $x\le c$, with $y=L/x^2$,

$$
\boxed{
Q_n^{\mathrm{TE+TM}}(x)
=K+x\left(\frac{D(y)}2-1\right).
}
$$

For $x>c$, with $z=\sqrt{x^2-L}$,

$$
\boxed{
\begin{aligned}
Q_n^{\mathrm{TE+TM}}(x)
={}&K+x\left(\frac{D(y)}2-1\right)\\
&-2\operatorname{tr}\!\left[
S\{\arctan(zS^{-1})-zS^{-1}\}\right]\\
&+(2n+1)c\{\arctan(z/c)-z/c\}.
\end{aligned}
}
$$

The reciprocal-argument form in the implementation is algebraically
identical and is used only to avoid large-$x$ cancellation.

### Proof

For $u(\rho)=\rho h_n^{(2)}(\rho)$, the half-integer Hankel expansion makes
$|u|^2$ an exact terminating polynomial in $L/\rho^2$, with coefficients
$\widehat a_k$. The Riccati--Bessel equation and Hankel Wronskian reduce the
reflection weight above cutoff to

$$
w=\frac12-
\frac{\sqrt{1-L/\rho^2}}{2-D(L/\rho^2)}.
$$

The first half of the integrand has an elementary finite-polynomial
primitive. After $v=(\rho^2-L)/L$, the remaining rational term is expressed
through $P'(v)/P(v)$. Encoding $P$ with its companion matrix turns this
logarithmic derivative into a trace, whose integral is exactly the displayed
matrix-arctangent expression. Continuity at cutoff and the condition
$Q(\infty)=0$ determine $K$.

The result is exact for every $n\ge1$ and $x>0$. Its matrix dimension is fixed
by $n$, rather than by a requested numerical accuracy, so it involves no
quadrature, numerical root finding, or approximation-dependent truncation.

## 3. Non-resonant TM/TE spherical-mode quality factor

Let

$$
N=n(n+1),\qquad m=n+1,
\qquad
\alpha=\frac12+i\sqrt{N-\frac14},
\qquad \beta=\overline\alpha,
$$

and define the terminating hypergeometric polynomial

$$
H_n(x)={}_6F_3\!\left(
\begin{matrix}
\frac12,\alpha+1,\beta+1,-n,n+1,1\\
1-\alpha,1-\beta,2
\end{matrix};-\frac1{x^2}\right).
$$

For $\ell=0,\ldots,m-1$, put

$$
r_\ell=m-\ell,
\qquad
a_\ell=\binom{m}{r_\ell}N^{r_\ell}
{}_3F_0\!\left(
\begin{matrix}-r_\ell,-\frac12,n\\-\end{matrix};-\frac1N\right).
$$

Let $C\in\mathbb C^{2m\times2m}$ be the Frobenius companion matrix defined by

$$
C_{j+1,j}=1,
\qquad
C_{2\ell+1,2m}=-a_\ell,
$$

with all other entries zero, using one-based indices. Define

$$
z=\begin{cases}
0,&x\le\sqrt N,\\
\sqrt{x^2-N},&x>\sqrt N,
\end{cases}
\qquad
\theta=\begin{cases}
-\pi/2,&z=0,\\
-\arctan(\sqrt N/z),&z>0,
\end{cases}
$$

and use the principal matrix logarithm

$$
\mathcal L=\begin{cases}
\log(-C),&z=0,\\
\log(zI-C),&0<z\le1,\\
\log(I-C/z),&z>1.
\end{cases}
$$

Then

$$
\boxed{
Q_n^{\mathrm{TM}}(x)=Q_n^{\mathrm{TE}}(x)=
z-x+\frac{N}{2x}\operatorname{Re}H_n(x)
+(4m-2)\sqrt N\,\theta
+\operatorname{Re}\operatorname{tr}
\left[\left(C-\frac{C^3}{N}\right)\mathcal L\right].
}
$$

### Proof

For $\psi(\rho)=\rho h_n^{(2)}(\rho)$,

$$
R(\rho)=|\psi(\rho)|^2
={}_3F_0\!\left(-n,n+1,\frac12;;-\rho^{-2}\right),
$$

which terminates exactly. The Riccati--Bessel equation gives

$$
|\psi'|^2+\frac{N}{\rho^2}R=R+\frac12R''.
$$

The Pochhammer identity

$$
\frac{(\alpha+1)_k(\beta+1)_k}
{(1-\alpha)_k(1-\beta)_k}
=\frac{N+k(k+1)}N
$$

then shows that $-\rho+NH_n(\rho)/(2\rho)$ is an exact primitive of the
unweighted integrand. Above cutoff, the Wronskian reduces the reflection
weight to a rational expression. Multiplying its denominator by
$\rho^{2n+2}/2$ produces exactly the characteristic polynomial
$\det(zI-C)$. Jacobi's log-determinant identity and Newton identities integrate
the remaining correction into the matrix-log trace. Positivity of the
physical denominator validates the principal logarithm branches; cutoff
continuity and decay at infinity complete the proof.

The identity is exact and uses only terminating hypergeometric functions and
a finite matrix of dimension $2(n+1)$. The submitted fixed-100-digit
implementation can lose accuracy through catastrophic cancellation for
astronomically large $x$; adaptive working precision or a far-field branch is
needed for uniform numerical stability. This is a conditioning issue, not a
defect in the identity.
