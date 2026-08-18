#!/usr/bin/env python3
"""Build the final GPT-5.6 Pro solution report after Terra compliance review."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any

from baseline_comparator import compare_against_baseline, load_baselines


DEFAULT_ARCHIVE = Path("openai_gpt-5.6-sol_20260725_195220.zip")
TERRA_FILENAME = (
    "openai_gpt-5.6-sol_20260725_195220_compliance_"
    "gpt-5.6-terra_medium_integrated-rubric_5trials.jsonl"
)

TITLE_OVERRIDES = {
    "spinor_norm_integral_i0": "Spinor Norm Integral $I_0$",
    "spherical_mode_quality_factor_te_tm": (
        "Equal-Power TE+TM Spherical-Mode Quality Factor"
    ),
    "spherical_mode_quality_factor_tm_te": (
        "Non-Resonant TM/TE Spherical-Mode Quality Factor"
    ),
    "monomer_dimer_entropy": "Monomer–Dimer Entropy Constant",
    "sextic_freud_moment_mu2": (
        "Second Moment of the Symmetric Sextic Freud Weight"
    ),
}


PRE_EXISTING_CERTIFICATES = {
    "autocorr_upper": {
        "classification": "pre_existing_certificate",
        "is_model_originated": False,
        "counted_in_score": False,
        "certificate_path": (
            "reports/gpt56_pro_final_solutions/certificates/"
            "autocorr_upper.json"
        ),
        "certificate_sha256": (
            "a2d2c953704be161f34a421269464ba9e48ba0fe17a4fd81ff0fd69b26d70d80"
        ),
        "reproducibility": "fixed_certificate_archived_separately",
        "objective_variant": "nonnegative_one_sided_maximum",
        "benchmark_value": 1.5028503020710076,
        "audit_note": (
            "Previously available fixed certificate reproduced for "
            "benchmark verification; excluded from model-originated "
            "findings."
        ),
    },
    "autocorr_signed_upper": {
        "classification": "pre_existing_certificate",
        "is_model_originated": False,
        "counted_in_score": False,
        "certificate_path": (
            "reports/gpt56_pro_final_solutions/certificates/"
            "autocorr_signed_upper.json"
        ),
        "certificate_sha256": (
            "0e86498ba294fb7a45606e3b8aa62765830fe13ed0539c8e57a8e8b0e49c9fae"
        ),
        "reproducibility": "fixed_certificate_archived_separately",
        "objective_variant": "signed_one_sided_maximum",
        "benchmark_value": 1.4545548626983325,
        "audit_note": (
            "Previously available fixed certificate reproduced for "
            "benchmark verification; excluded from model-originated "
            "findings."
        ),
    },
}


def accepted_solution_classification(problem_id: str, tier: int) -> str:
    """Classify an accepted output after provenance review."""
    if problem_id in PRE_EXISTING_CERTIFICATES:
        return "pre_existing_certificate"
    if tier > 0:
        return "new_solution"
    return "tier_0_calibration"


def verify_fixed_certificates(root: Path) -> None:
    """Fail report generation if a frozen certificate is missing or changed."""
    for problem_id, review in PRE_EXISTING_CERTIFICATES.items():
        path = root / review["certificate_path"]
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing fixed certificate for {problem_id}: {path}"
            )
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        expected = review["certificate_sha256"]
        if actual != expected:
            raise ValueError(
                f"Certificate hash mismatch for {problem_id}: "
                f"expected {expected}, got {actual}"
            )

MATHEMATICAL_RENDERINGS = {
    "w4_watson_integral": r"""
\[
\boxed{\displaystyle
W_4=
\frac{3g^9}{64\pi^6}
{}_4F_3\!\left(
\begin{matrix}\frac16,\frac13,\frac13,\frac12\\
\frac23,\frac56,\frac56\end{matrix};1\right)
-
\frac{16\sqrt3\,\pi^3}{3g^9}
{}_4F_3\!\left(
\begin{matrix}\frac12,\frac23,\frac23,\frac56\\
\frac76,\frac76,\frac43\end{matrix};1\right),
\qquad g=\Gamma\!\left(\frac13\right).
}
\]
""",
    "elliptic_k_moment_3": r"""
\[
\boxed{\displaystyle
\int_0^1 K(k)^3\,dk
=\frac{3\,\Gamma\!\left(\frac14\right)^8}{1280\pi^2}.}
\]
""",
    "elliptic_k2_e_moment": r"""
\[
\boxed{\displaystyle
\int_0^1 K(k)^2E(k)\,dk
=\frac{\Gamma\!\left(\frac14\right)^8}{640\pi^2}.}
\]
""",
    "airy_moment_a4": r"""
\[
\boxed{\displaystyle
a_4=\int_0^\infty \operatorname{Ai}(x)^4\,dx
=\frac{\log 3}{24\pi^2}.}
\]
""",
    "central_binomial_s5": r"""
\[
\omega=e^{2\pi i/3}=\frac{-1+i\sqrt3}{2},\qquad
\boxed{\displaystyle
S_5=
\frac{9\pi}{4}\operatorname{Im}\operatorname{Li}_4(\omega)
+\frac{\pi^2\zeta(3)}9-\frac{19\zeta(5)}3.}
\]
""",
    "airy_moment_a5": r"""
Define
\[
\mathcal H(x,y)=
\sum_{j,k\ge0}
\frac{(\frac23)_j(1)_j(\frac12)_k(\frac43)_{j+k}}
     {(\frac43)_j(\frac{11}{6})_{j+k}}
\frac{x^j}{j!}\frac{y^k}{k!}.
\]
Then
\[
\boxed{\displaystyle
a_5=
\frac{F_1\!\left(1;\frac13,\frac12;\frac32;
\frac1{16},\frac14\right)}
{24\pi^2\,3^{2/3}\Gamma(\frac23)}
-
\frac{\mathcal H(\frac1{16},\frac14)}
{3^{1/3}48^{4/3}\pi^{3/2}
\Gamma(\frac13)\Gamma(\frac{11}{6})}.}
\]
""",
    "mzv_reduction_zeta_3_3_3": r"""
\[
\boxed{\displaystyle
\zeta(3,3,3)=
\frac{\zeta(3)^3-3\zeta(3)\zeta(6)+2\zeta(9)}6.}
\]
""",
    "mahler_x_3_y_3_1_5xy": r"""
\[
\boxed{\displaystyle
m(x^3+y^3+1-5xy)
=\log 5-\frac{2}{125}
{}_4F_3\!\left(
\begin{matrix}1,1,\frac43,\frac53\\2,2,2\end{matrix};
\frac{27}{125}\right).}
\]
""",
    "saw_triangular_lattice": r"""
Let
\[
\mu_h=\sqrt{2+\sqrt2},\qquad q=2-\mu_h.
\]
The submitted conjecture is
\[
\boxed{\displaystyle
\mu_{\triangle}
=6-\mu_h-
\frac{2q^3(1+q^2)}
{5(1+q^4-3q^5)}.}
\]
""",
    "madelung_nacl": r"""
Let
\[
\beta=\frac{\Gamma(\frac18)^2}{\Gamma(\frac14)},\qquad
b=(2\sqrt2-2)^{1/4},\qquad
d=\sqrt{\frac{\sqrt3+1}{\sqrt8}}.
\]
Define
\[
C=-\frac18+\frac1{2\sqrt2}-\frac{4\pi}{3}
-\frac{\log2}{4\pi}
+\frac{\sqrt{2\sqrt2-2}\,\beta}{2\pi},
\]
\[
A=4\pi+\frac92\log(\sqrt2-1)
-6\log(2^{1/4}+1)-\frac{45}{8}\log2,
\]
\[
F=8\pi+2\sqrt2\log\!\left(
\frac{4(1-b)^2\sqrt{2b(1+b^2)}}{(1+b)^4}\right)
+6\sqrt2\log\!\left(
\frac{2^{3/4}(1+b)^2\beta}{64\pi}\right),
\]
\[
D=\frac{16\pi}{3}+\frac{4\sqrt3}{9}
\log\!\left(
\frac{(1-d)^4}
{32(1+d)^2\sqrt{2d(1+d^2)}}\right).
\]
The submitted NaCl Madelung expression is
\[
\boxed{\displaystyle M_{\mathrm{NaCl}}=-(C+A+F+D).}
\]
""",
    "spinor_norm_integral_i0": r"""
\[
\boxed{\displaystyle
I_0=
\frac{\Gamma(\frac34)^2+\frac18\Gamma(\frac14)^2}
{\sqrt\pi}.}
\]
""",
    "spherical_mode_quality_factor_te_tm": r"""
\[
L=n(n+1),\qquad c=\sqrt L,\qquad m=n+1.
\]
\[
\widehat a_0=1,\qquad
\widehat a_{k+1}=
\widehat a_k\,
\frac{(n+k+1)(n-k)(2k+1)}{2(k+1)L}.
\]
\[
s_0=2,\qquad s_{k+1}=-\frac{\widehat a_k}{k+1},\qquad
P(v)=\sum_{j=0}^{m}s_j(1+v)^{m-j}.
\]
\[
A=\operatorname{Companion}\!\left(
\frac{(-1)^mP(-u)}{[v^m]P(v)}\right),\qquad
S=c\sqrt A,\qquad
\operatorname{Re}\operatorname{tr}S\ge0.
\]
\[
y=\frac{L}{x^2},\quad
D=\sum_{k=0}^n\frac{\widehat a_k y^{k+1}}{k+1},\quad
D_2=\sum_{k=2}^n\frac{\widehat a_k y^{k+1}}{k+1},
\]
\[
K=\pi\!\left(\operatorname{tr}S-\frac{(2n+1)c}{2}\right).
\]
\[
\boxed{\displaystyle
Q_n^{\mathrm{TE+TM}}(x)=K+x\left(\frac D2-1\right),
\qquad x\le c.}
\]
\[
z=\sqrt{x^2-L},\qquad q=\frac zx,\qquad
\delta=\frac{y}{1+q},\qquad x>c.
\]
\[
\boxed{\displaystyle
Q_n^{\mathrm{TE+TM}}(x)=
\begin{cases}
K+x(\frac D2-1)
-2\operatorname{tr}\!\left[S\{\arctan(zS^{-1})-zS^{-1}\}\right]
+(2n+1)c\{\arctan(z/c)-z/c\},
&z\le |\operatorname{tr}S|/m,\\[2mm]
\frac{x}{2}\!\left[D_2-\delta^3(1-\frac{\delta}{4})\right]
+2\operatorname{tr}\!\left[S\{\arctan(S/z)-S/z+(S/z)^3/3\}\right]\\
\qquad -(2n+1)c\{\arctan(c/z)-c/z+(c/z)^3/3\},
&z>|\operatorname{tr}S|/m.
\end{cases}}
\]
""",
    "spherical_mode_quality_factor_tm_te": r"""
Let
\[
N=n(n+1),\quad m=n+1,\quad
\alpha=\frac12+i\sqrt{N-\frac14},\quad
\beta=\overline\alpha,
\]
\[
H_n(x)={}_6F_3\!\left(
\begin{matrix}\frac12,\alpha+1,\beta+1,-n,n+1,1\\
1-\alpha,1-\beta,2\end{matrix};-\frac1{x^2}\right).
\]
\[
r_\ell=m-\ell,\qquad
a_\ell=\binom{m}{r_\ell}N^{r_\ell}
{}_3F_0\!\left(
\begin{matrix}-r_\ell,-\frac12,n\\-\end{matrix};-\frac1N\right).
\]
\[
C\in\mathbb C^{2m\times2m},\qquad
C_{j+1,j}=1,\qquad C_{2\ell+1,\,2m}=-a_\ell,
\qquad C_{ij}=0\ \text{otherwise},
\]
\[
z=\begin{cases}0,&x\le\sqrt N,\\ \sqrt{x^2-N},&x>\sqrt N,\end{cases}
\quad
\theta=\begin{cases}-\pi/2,&z=0,\\-\arctan(\sqrt N/z),&z>0,\end{cases}
\]
\[
\mathcal L=\begin{cases}
\log(-C),&z=0,\\
\log(zI-C),&0<z\le1,\\
\log(I-C/z),&z>1.
\end{cases}
\]
\[
\boxed{\displaystyle
Q_n^{\mathrm{TM}}(x)=Q_n^{\mathrm{TE}}(x)=
z-x+\frac{N}{2x}\operatorname{Re}H_n(x)
+(4m-2)\sqrt N\,\theta
+\operatorname{Re}\operatorname{tr}
\!\left[\left(C-\frac{C^3}{N}\right)\mathcal L\right].}
\]
""",
    "autocorr_upper": r"""
\[
\mathbf v=(v_0,\ldots,v_{89999})\in\mathbb R_{\ge0}^{90000},
\qquad
f(x)=v_j\quad\text{for}\quad
-\frac14+\frac{j}{180000}\le x<
-\frac14+\frac{j+1}{180000}.
\]
\[
\boxed{\displaystyle
\frac{\max_t(f*f)(t)}{\left(\int f\right)^2}
=1.5028503020710076.}
\]
""",
    "autocorr_signed_upper": r"""
\[
\mathbf v=(v_0,\ldots,v_{399})\in\mathbb R^{400},\qquad
f(x)=v_j\quad\text{on the \(j\)-th equal subinterval of }
\left[-\frac14,\frac14\right].
\]
\[
\boxed{\displaystyle
C'_{\mathrm{one\text{-}sided}}(\mathbf v)=
\frac{800\,\max_k(\mathbf v*\mathbf v)_k}
{\left(\sum_{j=0}^{399}v_j\right)^2}
=1.4545548626983325.}
\]
""",
    "keich_thin_triangles_128": r"""
\[
\boxed{\displaystyle
b_i=\frac{q_i}{1024},\qquad i=0,\ldots,127,}
\]
\[
R_i=\left\{(x,y):0\le x\le1,\;
\frac{i}{128}x+b_i-\frac{1-x}{128}
\le y\le \frac{i}{128}x+b_i\right\}.
\]
\[
\boxed{\displaystyle
\operatorname{Area}\!\left(\bigcup_{i=0}^{127}R_i\right)
=0.10914798918224512
<0.1148103258186177.}
\]
""",
    "ramsey_asymptotic": r"""
The submitted correction polynomial is
\[
p(\lambda)=-0.25\lambda+0.033\lambda^2
+0.08\lambda^3-0.0778\lambda^5,
\]
and hence
\[
F(\lambda)=(1+\lambda)\log(1+\lambda)
-\lambda\log\lambda+p(\lambda)e^{-\lambda}.
\]
It supplies 200 explicitly defined step intervals for \(M(\lambda)\) and
\[
Y_j=(1-0.0012)
\min\!\left(1,\frac{\frac14e^{0.137/e}}{X(\lambda_j,M_j)}\right),
\]
using the breakpoints and grid levels retained in the JSON companion. The
interval-arithmetic certificate proves all required inequalities and yields
\[
\boxed{\displaystyle
R(k,k)\le
\left(3.6960839126332994\right)^{k+o(k)},
\qquad 3.6960839126332994<3.7992.}
\]
""",
    "resultant_chebyshev": r"""
Let
\[
\begin{aligned}
P={}&51825539\cdot436089807149109873239\\
&{}\cdot13200334028406359184273669777594156498017041\\
&{}\cdot119009966720120470199067238863340312027302040429788911464472185079442597334859473441.
\end{aligned}
\]
The submitted numerical expression is
\[
\boxed{\displaystyle R=\frac{P^2}{2^{540}}.}
\]
""",
    "feigenbaum_delta": r"""
With
\[
N=
2180144366644995730246854765914024575250775656756443994447602214234689549111246771289477728250363532602148740597,
\]
the submission returns
\[
\boxed{\displaystyle \delta=\frac{\sqrt N}{10^{55}}.}
\]
""",
    "feigenbaum_alpha": r"""
The submitted value is the finite generalized continued fraction
\[
\boxed{\displaystyle
\alpha=
[2;1,1,85,2,8,1,10,16,3,8,9,2,1,40,\sqrt2].}
\]
Equivalently, the final tail is \(40+1/\sqrt2\), nested inside the
displayed sequence of partial quotients.
""",
    "nested_radical_kasner": r"""
\[
\boxed{\displaystyle
K=
\frac{
175793275661800453270881963821813852765319992214683770431013550038511023267444675757234455400025945297095
}{10^{104}+1}.}
\]
""",
    "stieltjes_gamma_1": r"""
Writing \(\zeta''(0,1)\) for the second derivative of the Hurwitz zeta
function with respect to its first argument, the submission is
\[
\boxed{\displaystyle
\gamma_1=
\zeta''(0,1)-\frac{\gamma^2}{2}
+\frac{\pi^2}{24}
+\frac{\log^2(2\pi)}2.}
\]
""",
    "euler_mascheroni_closed_form": r"""
\[
\boxed{\displaystyle
\gamma=
{}_2F_2\!\left(
\begin{matrix}1,1\\2,2\end{matrix};-1\right)
-e^{-1}U(1,1,1),}
\]
where \(U\) is Tricomi's confluent hypergeometric function.
""",
    "calabi_yau_c5": r"""
\[
\boxed{\displaystyle
C_5=
\frac{
95869411228790989677465668396217590140439479019447662973679749308496694302478578092951538171573178204361535269
}{10^{106}}.}
\]
""",
    "tracy_widom_f2_variance": r"""
\[
\boxed{\displaystyle
\operatorname{Var}(F_2)=
\frac{\pi^2}{12}
-\frac1{108}
-\frac1{77034}
-\frac1{19622790853}.}
\]
""",
    "elliptic_kernel_f2_001": r"""
\[
\boxed{\displaystyle
f_2(0,0,1)=
\frac{
307476526736391709896774235351358778861783865155459326024781812950213971132375910461620684439641407962420702403407811170933205901539809821596
}{10^{139}}.}
\]
""",
    "monomer_dimer_entropy": r"""
\[
\boxed{\displaystyle
h_{\mathrm{MD}}=
\frac{G}{\pi}+\frac{\log2}{2}
+\frac{4397789}{21716395}\frac{\zeta(3)}{\pi^2},}
\]
where \(G\) is Catalan's constant.
""",
    "saw_square_lattice": r"""
\[
\boxed{\displaystyle
\mu_{\square}=
\sqrt{\frac{7+\sqrt{30261}}{26}}
-\frac{7579\pi+14}{26\cdot581^5}.}
\]
""",
    "hard_square_entropy": r"""
\[
\boxed{\displaystyle
\kappa_{\mathrm{HS}}
=29310020811867649937^{\,1/110}.}
\]
""",
    "saw_simple_cubic": r"""
\[
\boxed{\displaystyle
\mu_{\mathrm{SC}}=
\sqrt{
22-\frac{1}{
\pi^2+2\pi+\gamma+\frac1{6(207-3)}
}}.}
\]
""",
    "madelung_zns": r"""
\[
\boxed{\displaystyle
M_{\mathrm{ZnS}}=
\frac{
1638055053388789423750034776358619465360179663136657883957644623927706812837223137698546420043494665161
}{10^{102}}.}
\]
""",
    "knot_volume_6_3": r"""
Define
\[
t=\left(\frac{9+\sqrt{93}}{18}\right)^{1/3}
-\left(\frac{\sqrt{93}-9}{18}\right)^{1/3},
\quad r=\sqrt{4-t^2},\quad
s=\sqrt{3+2t+3t^2},
\]
\[
u_0=\frac{2-t^2+itr}{2},\qquad
u_1=\frac{1-t+is}{1-t-is},\qquad
u_2=\frac{1+t+is}{1+t-is},\qquad
u_3=(u_1u_2)^{-1}.
\]
The submitted volume expression is
\[
\boxed{\displaystyle
\operatorname{Vol}(6_3)=
2\,\operatorname{Im}
\sum_{j=0}^{3}\operatorname{Li}_2(u_j).}
\]
""",
    "sextic_freud_moment_mu2": r"""
Put
\[
q=-\kappa\tau^2,\qquad X=\frac{\tau^3}{27},
\qquad Y=\frac{q^3}{27},
\]
and define
\[
\mathcal H(a;\mathbf b,\mathbf c;X,Y)=
\sum_{j,k\ge0}
\frac{(a)_{2j+k}}
{(b_1)_j(b_2)_j(c_1)_k(c_2)_k}
\frac{X^j}{j!}\frac{Y^k}{k!}.
\]
Let
\[
\mathbf b_0=\left(\frac13,\frac23\right),\quad
\mathbf b_1=\left(\frac23,\frac43\right),\quad
\mathbf b_2=\left(\frac43,\frac53\right).
\]
\[
\mathcal H(a;\mathbf b,\mathbf c)
:=\mathcal H(a;\mathbf b,\mathbf c;X,Y).
\]
The submitted expression is
\[
\boxed{\displaystyle
\begin{aligned}
\mu_2(\tau,\kappa)=\frac13\bigg[&
\Gamma(\tfrac12)\mathcal H(\tfrac12;\mathbf b_0,\mathbf b_0)
+q\Gamma(\tfrac56)\mathcal H(\tfrac56;\mathbf b_0,\mathbf b_1)\\
&+\frac{q^2}{2}\Gamma(\tfrac76)\mathcal H(\tfrac76;\mathbf b_0,\mathbf b_2)
+\tau\Gamma(\tfrac76)\mathcal H(\tfrac76;\mathbf b_1,\mathbf b_0)\\
&+\tau q\Gamma(\tfrac32)\mathcal H(\tfrac32;\mathbf b_1,\mathbf b_1)
+\frac{\tau q^2}{2}\Gamma(\tfrac{11}{6})\mathcal H(\tfrac{11}{6};\mathbf b_1,\mathbf b_2)\\
&+\frac{\tau^2}{2}\Gamma(\tfrac{11}{6})\mathcal H(\tfrac{11}{6};\mathbf b_2,\mathbf b_0)
+\frac{\tau^2q}{2}\Gamma(\tfrac{13}{6})\mathcal H(\tfrac{13}{6};\mathbf b_2,\mathbf b_1)\\
&+\frac{\tau^2q^2}{4}\Gamma(\tfrac52)\mathcal H(\tfrac52;\mathbf b_2,\mathbf b_2)
\bigg],
\end{aligned}}
\]
where every \(\mathcal H\) has the common arguments \((X,Y)\).
""",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]


def read_archive_jsonl(
    archive: Path, filename: str
) -> list[dict[str, Any]]:
    with zipfile.ZipFile(archive) as zf:
        member = next(
            name for name in zf.namelist() if name.endswith(f"/{filename}")
        )
        return [
            json.loads(line)
            for line in zf.read(member).decode().splitlines()
            if line.strip()
        ]


def read_archive_json(archive: Path, filename: str) -> dict[str, Any]:
    with zipfile.ZipFile(archive) as zf:
        member = next(
            name for name in zf.namelist() if name.endswith(f"/{filename}")
        )
        return json.loads(zf.read(member).decode())


def problem_title(prompt: str, problem_id: str) -> str:
    if problem_id in TITLE_OVERRIDES:
        return TITLE_OVERRIDES[problem_id]

    first_line = next(
        (line.strip() for line in prompt.splitlines() if line.strip()), ""
    )
    heading = re.match(r"^#{1,6}\s+(.+?)\s*$", first_line)
    if heading:
        return heading.group(1).strip()

    generic_prefixes = (
        "definition",
        "goal",
        "task",
        "current state-of-the-art",
        "required output format",
        "inadmissible",
        "constraints",
        "to beat",
        "upper edge",
        "lower edge",
        "vertical edge",
    )
    for match in re.finditer(r"\*\*([^*\n]+)\*\*", prompt[:1200]):
        candidate = match.group(1).strip()
        if not candidate.lower().startswith(generic_prefixes):
            return candidate
    return problem_id.replace("_", " ")


def benchmark_evidence(evaluation: dict[str, Any]) -> str:
    metrics = evaluation["validator_metrics"]
    problem_id = evaluation["problem_id"]
    if problem_id == "autocorr_upper":
        return (
            f"Autoconvolution ratio {metrics['autoconvolution_ratio']:.15f} "
            f"matches frozen benchmark {metrics['best_known_upper']}"
        )
    if problem_id == "autocorr_signed_upper":
        return (
            f"One-sided signed-autoconvolution ratio "
            f"{metrics['autoconvolution_ratio']:.15f} "
            f"matches frozen benchmark {metrics['best_known_upper']}"
        )
    if problem_id == "keich_thin_triangles_128":
        baseline = evaluation["baseline_comparison"]["baseline_value"]
        return (
            f"Union area {metrics['area']:.15f} "
            f"< {baseline:.15f}"
        )
    if problem_id == "ramsey_asymptotic":
        baseline = evaluation["baseline_comparison"]["baseline_value"]
        return (
            f"Certified Ramsey growth base "
            f"{metrics['growth_base_c']:.12f} < {baseline}"
        )
    return evaluation["validator_message"]


def markdown_escape(value: Any) -> str:
    return str(value).replace("|", r"\|").replace("\n", " ")


def render_submitted_response(response: str) -> str:
    match = re.search(
        r"```(?:python)?[ \t]*\n(.*?)```", response, re.DOTALL
    )
    code = match.group(1).strip() if match else response.strip()
    code = "\n".join(line.rstrip() for line in code.splitlines())
    return f"```python\n{code}\n```"


def markdown_math(value: str) -> str:
    """Use math delimiters supported by common Markdown renderers."""
    value = re.sub(r"(?m)^[ \t]*\\\[[ \t]*$", "$$", value)
    value = re.sub(r"(?m)^[ \t]*\\\][ \t]*$", "$$", value)
    return value.replace(r"\(", "$").replace(r"\)", "$")


def final_answer_markdown(value: str) -> str:
    """Keep only display equations; prose belongs in the data fields."""
    normalized = markdown_math(value)
    blocks: list[str] = []
    current: list[str] | None = None
    for line in normalized.splitlines():
        if line.strip() == "$$":
            if current is None:
                current = ["$$"]
            else:
                current.append("$$")
                blocks.append("\n".join(current))
                current = None
        elif current is not None:
            current.append(line)
    if current is not None:
        raise ValueError("Unbalanced display-math delimiters")
    return "\n\n".join(blocks)


def build_report(root: Path, archive: Path) -> tuple[dict[str, Any], str]:
    verify_fixed_certificates(root)
    problems = json.loads((root / "data/problems_full.json").read_text())
    problem_by_id = {problem["id"]: problem for problem in problems}

    responses = {
        row["problem_id"]: row
        for row in read_archive_jsonl(archive, "responses.jsonl")
    }
    evaluations = {
        row["problem_id"]: row
        for row in read_archive_jsonl(archive, "evaluation.jsonl")
    }
    run_config = read_archive_json(archive, "config.json")
    original_summary = read_archive_json(archive, "summary.json")
    baselines = load_baselines(root / "data/baselines.json")

    terra_rows = read_jsonl(root / "results" / TERRA_FILENAME)
    terra_by_id = {row["problem_id"]: row for row in terra_rows}
    manual_rows = json.loads(
        (
            root
            / "reports/compliance_checker_study/manual_adjudication.json"
        ).read_text()
    )
    manual_by_id = {row["problem_id"]: row for row in manual_rows}

    old_summary_rows = list(
        csv.DictReader(
            (
                root
                / "reports/compliance_checker_study/reviewer_summary.csv"
            ).open()
        )
    )
    old_gemini = next(
        row
        for row in old_summary_rows
        if row["reviewer"] == "Gemini 3 Flash (low, baseline rubric)"
    )

    numeric_accepted = {
        problem_id
        for problem_id, row in terra_by_id.items()
        if row["compliant"]
    }
    benchmark_accepted = {
        problem_id
        for problem_id, row in evaluations.items()
        if row.get("mode") == "benchmark"
        and row.get("valid")
        and row.get("baseline_comparison", {}).get("result")
        == "beats_baseline"
    }
    accepted_ids = numeric_accepted | benchmark_accepted

    solutions: list[dict[str, Any]] = []
    for problem_id in sorted(
        accepted_ids,
        key=lambda item: (
            0 if problem_by_id[item]["solvability"] > 0 else 1,
            problem_by_id[item]["solvability"],
            evaluations[item]["problem_index"],
        ),
    ):
        problem = problem_by_id[problem_id]
        evaluation = evaluations[problem_id]
        tier = int(problem["solvability"])
        classification = accepted_solution_classification(problem_id, tier)
        is_new = classification == "new_solution"
        counted_in_score = classification != "pre_existing_certificate"

        if problem_id in terra_by_id:
            terra = terra_by_id[problem_id]
            manual = manual_by_id[problem_id]
            verification = {
                "type": "numeric_then_compliance",
                "numeric_accuracy_passed": bool(evaluation["success"]),
                "matching_digits": evaluation.get("matching_digits"),
                "terra": {
                    "model": terra["model"],
                    "reasoning_effort": terra["reasoning_effort"],
                    "rubric": "integrated",
                    "compliant": terra["compliant"],
                    "compliant_votes": terra["compliant_votes"],
                    "total_valid_votes": terra["total_valid_votes"],
                    "backend_verified": terra["backend_verified"],
                    "judgments": [
                        {
                            "trial": index,
                            "compliant": round_row["compliant"],
                            "reason": round_row["reason"],
                        }
                        for index, round_row in enumerate(
                            terra["rounds"], start=1
                        )
                    ],
                },
                "adjudication": {
                    "confidence": manual["confidence"],
                    "rules": manual["rules"],
                    "rationale": manual["rationale"],
                },
            }
            verification_summary = (
                f"Numeric gate passed; Terra "
                f"{terra['compliant_votes']}/{terra['total_valid_votes']} pass"
            )
        else:
            validator_metrics = dict(evaluation["validator_metrics"])
            validator_message = evaluation["validator_message"]
            baseline_comparison = evaluation["baseline_comparison"]
            if problem_id in PRE_EXISTING_CERTIFICATES:
                benchmark_value = PRE_EXISTING_CERTIFICATES[problem_id][
                    "benchmark_value"
                ]
                validator_metrics["best_known_upper"] = benchmark_value
                validator_metrics["improves_bound"] = False
                baseline_comparison = compare_against_baseline(
                    problem_id, validator_metrics, baselines
                ).to_dict()
                validator_message = (
                    "Fixed certificate reproduces the frozen benchmark "
                    f"value {benchmark_value}."
                )

            normalized_evaluation = {
                **evaluation,
                "validator_metrics": validator_metrics,
                "validator_message": validator_message,
                "baseline_comparison": baseline_comparison,
            }
            verification = {
                "type": "deterministic_benchmark_validator",
                "valid": bool(evaluation["valid"]),
                "validator_message": validator_message,
                "validator_metrics": validator_metrics,
                "baseline_comparison": baseline_comparison,
            }
            verification_summary = benchmark_evidence(
                normalized_evaluation
            )

        solutions.append(
            {
                "problem_id": problem_id,
                "title": problem_title(problem["prompt"], problem_id),
                "tier": tier,
                "classification": classification,
                "is_new_solution": is_new,
                "counted_in_score": counted_in_score,
                "provenance_review": PRE_EXISTING_CERTIFICATES.get(
                    problem_id
                ),
                "domain": problem["domain"],
                "output_type": problem["output_type"],
                "evaluation_mode": problem["evaluation_mode"],
                "verification_summary": verification_summary,
                "verification": verification,
                "mathematical_rendering": final_answer_markdown(
                    MATHEMATICAL_RENDERINGS[problem_id].strip()
                ),
                "source_url": problem.get("source_url"),
                "source_note": problem.get("source_note"),
                "submitted_response": responses[problem_id]["response"],
                "response_metadata": {
                    "provider": responses[problem_id]["provider"],
                    "model": responses[problem_id]["model"],
                    "reasoning_effort": responses[problem_id].get(
                        "reasoning_effort"
                    ),
                    "timestamp": responses[problem_id].get("timestamp"),
                },
            }
        )

    rejected_solutions: list[dict[str, Any]] = []
    for terra in sorted(
        (row for row in terra_rows if not row["compliant"]),
        key=lambda row: evaluations[row["problem_id"]]["problem_index"],
    ):
        problem_id = terra["problem_id"]
        problem = problem_by_id[problem_id]
        evaluation = evaluations[problem_id]
        manual = manual_by_id[problem_id]
        rejected_solutions.append(
            {
                "problem_id": problem_id,
                "title": problem_title(problem["prompt"], problem_id),
                "tier": int(problem["solvability"]),
                "classification": (
                    "numerically_correct_but_permissibility_rejected"
                ),
                "is_final_pass": False,
                "domain": problem["domain"],
                "output_type": problem["output_type"],
                "evaluation_mode": problem["evaluation_mode"],
                "numeric_accuracy": {
                    "passed": bool(evaluation["success"]),
                    "matching_digits": evaluation.get("matching_digits"),
                },
                "terra": {
                    "model": terra["model"],
                    "reasoning_effort": terra["reasoning_effort"],
                    "rubric": "integrated",
                    "compliant": False,
                    "compliant_votes": terra["compliant_votes"],
                    "total_valid_votes": terra["total_valid_votes"],
                    "backend_verified": terra["backend_verified"],
                    "judgments": [
                        {
                            "trial": index,
                            "compliant": round_row["compliant"],
                            "reason": round_row["reason"],
                        }
                        for index, round_row in enumerate(
                            terra["rounds"], start=1
                        )
                    ],
                },
                "rejection": {
                    "confidence": manual["confidence"],
                    "rules": manual["rules"],
                    "rationale": manual["rationale"],
                },
                "mathematical_rendering": final_answer_markdown(
                    MATHEMATICAL_RENDERINGS[problem_id].strip()
                ),
                "source_url": problem.get("source_url"),
                "source_note": problem.get("source_note"),
                "submitted_response": responses[problem_id]["response"],
                "response_metadata": {
                    "provider": responses[problem_id]["provider"],
                    "model": responses[problem_id]["model"],
                    "reasoning_effort": responses[problem_id].get(
                        "reasoning_effort"
                    ),
                    "timestamp": responses[problem_id].get("timestamp"),
                },
            }
        )

    accepted_outputs = solutions
    external_certificates = [
        row
        for row in accepted_outputs
        if row["classification"] == "pre_existing_certificate"
    ]
    for row in external_certificates:
        row.pop("submitted_response")
        row["original_submission_retained_in_source_archive"] = True
    solutions = [
        row for row in accepted_outputs if row["counted_in_score"]
    ]

    total_problems = len(problems)
    raw_accepted = len(accepted_outputs)
    final_passed = len(solutions)
    new_solutions = [row for row in solutions if row["is_new_solution"]]
    calibration_solutions = [
        row
        for row in solutions
        if row["classification"] == "tier_0_calibration"
    ]
    tier_totals = Counter(int(problem["solvability"]) for problem in problems)
    tier_passes = Counter(row["tier"] for row in solutions)
    tier_raw_accepted = Counter(row["tier"] for row in accepted_outputs)
    tier_external = Counter(row["tier"] for row in external_certificates)
    tier_new = Counter(row["tier"] for row in new_solutions)
    mode_totals = Counter(problem["evaluation_mode"] for problem in problems)
    mode_passes = Counter(row["evaluation_mode"] for row in solutions)
    mode_raw_accepted = Counter(
        row["evaluation_mode"] for row in accepted_outputs
    )
    scored_benchmark_accepted = benchmark_accepted.difference(
        PRE_EXISTING_CERTIFICATES
    )

    majority_aligned_votes = sum(
        max(row["compliant_votes"], 5 - row["compliant_votes"])
        for row in terra_rows
    )
    terra_unanimous_pass = sum(
        row["compliant_votes"] == 5 for row in terra_rows
    )
    terra_unanimous_reject = sum(
        row["compliant_votes"] == 0 for row in terra_rows
    )

    artifact = {
        "schema_version": "1.2",
        "report_title": (
            "GPT-5.6 Pro (max reasoning) final solutions after "
            "permissibility review and autocorrelation-certificate "
            "provenance adjustment"
        ),
        "run": {
            "run_id": run_config["run_id"],
            "timestamp": run_config["timestamp"],
            "provider": run_config["provider"],
            "model": run_config["model"],
            "display_name": "GPT-5.6 Pro",
            "reasoning_effort": run_config["reasoning_effort"],
            "source_archive": archive.name,
        },
        "review_scope": {
            "provenance_adjustment": sorted(PRE_EXISTING_CERTIFICATES),
            "other_accepted_candidates_re_adjudicated": False,
        },
        "final_score": {
            "passed": final_passed,
            "total": total_problems,
            "failed": total_problems - final_passed,
            "pass_rate": round(final_passed / total_problems, 6),
            "pass_rate_percent": round(
                100 * final_passed / total_problems, 1
            ),
            "new_solutions_tiers_1_to_3": len(new_solutions),
            "tier_0_calibration_solutions": len(calibration_solutions),
            "raw_accepted_outputs": raw_accepted,
            "pre_existing_certificates_excluded": len(
                external_certificates
            ),
        },
        "pipeline": {
            "pre_compliance_raw_passes": original_summary["passed"],
            "numeric_accuracy_passes_sent_to_terra": len(terra_rows),
            "numeric_solutions_accepted_by_terra": len(numeric_accepted),
            "numeric_solutions_rejected_by_terra": (
                len(terra_rows) - len(numeric_accepted)
            ),
            "raw_deterministic_benchmark_acceptances": len(
                benchmark_accepted
            ),
            "deterministic_benchmark_improvements": len(
                scored_benchmark_accepted
            ),
            "raw_accepted_outputs": raw_accepted,
            "pre_existing_certificates_excluded": len(
                external_certificates
            ),
            "final_passes": final_passed,
        },
        "by_tier": {
            str(tier): {
                "passed": tier_passes[tier],
                "raw_accepted_outputs": tier_raw_accepted[tier],
                "pre_existing_certificates": tier_external[tier],
                "new_solutions": tier_new[tier],
                "total": tier_totals[tier],
                "pass_rate_percent": round(
                    100 * tier_passes[tier] / tier_totals[tier], 1
                ),
                "new_solution_tier": tier > 0,
            }
            for tier in sorted(tier_totals)
        },
        "by_evaluation_mode": {
            mode: {
                "passed": mode_passes[mode],
                "raw_accepted_outputs": mode_raw_accepted[mode],
                "total": mode_totals[mode],
                "pass_rate_percent": round(
                    100 * mode_passes[mode] / mode_totals[mode], 1
                ),
            }
            for mode in sorted(mode_totals)
        },
        "permissibility_checker_statistics": {
            "reviewer": "gpt-5.6-terra",
            "reasoning_effort": "medium",
            "rubric": "integrated",
            "trials_per_candidate": 5,
            "candidates": len(terra_rows),
            "valid_judgments": sum(
                row["total_valid_votes"] for row in terra_rows
            ),
            "accepted": len(numeric_accepted),
            "rejected": len(terra_rows) - len(numeric_accepted),
            "majority_aligned_votes": majority_aligned_votes,
            "majority_alignment_percent": round(
                100 * majority_aligned_votes / (5 * len(terra_rows)), 1
            ),
            "unanimous_acceptances": terra_unanimous_pass,
            "unanimous_rejections": terra_unanimous_reject,
            "non_unanimous_decisions": [
                {
                    "problem_id": row["problem_id"],
                    "votes": (
                        f"{row['compliant_votes']}/"
                        f"{row['total_valid_votes']}"
                    ),
                    "verdict": (
                        "pass" if row["compliant"] else "reject"
                    ),
                }
                for row in terra_rows
                if row["compliant_votes"] not in {0, 5}
            ],
            "agreement_with_reference_adjudication": {
                "correct": 29,
                "total": 29,
                "percent": 100.0,
                "false_accepts": 0,
                "false_rejects": 0,
            },
            "old_gemini_baseline": {
                "correct": 25,
                "total": 29,
                "agreement_percent": 86.2,
                "cohens_kappa": float(
                    old_gemini["cohens_kappa_with_codex_manual"]
                ),
                "false_accepts": int(old_gemini["false_accept"]),
                "false_rejects": int(old_gemini["false_reject"]),
            },
        },
        "solutions": solutions,
        "pre_existing_certificates": external_certificates,
        "terra_rejected_numeric_candidates": rejected_solutions,
    }

    lines = [
        "# GPT-5.6 Pro final solutions",
        "",
        "**Model:** GPT-5.6 Pro (`gpt-5.6-sol`), max reasoning",
        "",
        f"**Run:** `{run_config['run_id']}`",
        "",
        (
            "**Score after excluding the two pre-existing "
            "autocorrelation certificates:** "
            f"**{final_passed}/{total_problems} "
            f"({100 * final_passed / total_problems:.1f}%)**"
        ),
        "",
        (
            "**Raw accepted outputs before provenance review:** "
            f"**{raw_accepted}/{total_problems} "
            f"({100 * raw_accepted / total_problems:.1f}%)**"
        ),
        "",
        (
            "**New benchmark solutions under this limited review "
            "(Tiers 1–3):** "
            f"**{len(new_solutions)}**"
        ),
        "",
        (
            "**Pre-existing certificates excluded from score:** "
            f"**{len(external_certificates)}**"
        ),
        "",
        (
            "_Scope: this provenance adjustment covers only the two "
            "autocorrelation certificates; other accepted candidates "
            "retain their prior report status._"
        ),
        "",
        "## Final statistics",
        "",
        "| Stage | Passed | Total | Rate |",
        "|---|---:|---:|---:|",
        (
            f"| Original evaluator, before permissibility filtering | "
            f"{original_summary['passed']} | {total_problems} | "
            f"{100 * original_summary['passed'] / total_problems:.1f}% |"
        ),
        (
            f"| Numeric candidates accepted by Terra | "
            f"{len(numeric_accepted)} | {len(terra_rows)} | "
            f"{100 * len(numeric_accepted) / len(terra_rows):.1f}% |"
        ),
        (
            f"| Raw deterministic benchmark acceptances | "
            f"{len(benchmark_accepted)} | "
            f"{mode_totals['benchmark_best_known']} | "
            f"{100 * len(benchmark_accepted) / mode_totals['benchmark_best_known']:.1f}% |"
        ),
        (
            f"| Scored deterministic benchmark improvements | "
            f"{len(scored_benchmark_accepted)} | "
            f"{mode_totals['benchmark_best_known']} | "
            f"{100 * len(scored_benchmark_accepted) / mode_totals['benchmark_best_known']:.1f}% |"
        ),
        (
            f"| Raw accepted outputs | {raw_accepted} | "
            f"{total_problems} | "
            f"{100 * raw_accepted / total_problems:.1f}% |"
        ),
        (
            f"| **Autocorrelation-adjusted result** | **{final_passed}** | "
            f"**{total_problems}** | "
            f"**{100 * final_passed / total_problems:.1f}%** |"
        ),
        "",
        "### Score by tier",
        "",
        "| Tier | Status | Credited | Pre-existing | Total | Rate |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for tier in sorted(tier_totals):
        status = "Calibration" if tier == 0 else "**New solutions**"
        lines.append(
            f"| {tier} | {status} | {tier_passes[tier]} | "
            f"{tier_external[tier]} | "
            f"{tier_totals[tier]} | "
            f"{100 * tier_passes[tier] / tier_totals[tier]:.1f}% |"
        )

    lines.extend(
        [
            "",
            "### Terra five-trial permissibility statistics",
            "",
            "- 29 numerically correct candidates reviewed.",
            "- 145/145 valid checker judgments.",
            (
                f"- {len(numeric_accepted)} accepted and "
                f"{len(terra_rows) - len(numeric_accepted)} rejected."
            ),
            (
                f"- {majority_aligned_votes}/145 votes "
                f"({100 * majority_aligned_votes / 145:.1f}%) agreed "
                "with their candidate's final majority."
            ),
            (
                f"- {terra_unanimous_pass}/"
                f"{len(numeric_accepted)} acceptances and "
                f"{terra_unanimous_reject}/"
                f"{len(terra_rows) - len(numeric_accepted)} rejections "
                "were unanimous."
            ),
            (
                "- Majority decisions agreed with the reference "
                "adjudication on 29/29 candidates, with zero false "
                "accepts and zero false rejects."
            ),
            (
                "- The old Gemini baseline agreed on 25/29 candidates "
                "(86.2%, κ=0.721); the updated Terra protocol corrected "
                "its two false accepts and two false rejects."
            ),
            "",
            "## Evaluated-output index",
            "",
            "| Status | Problem | Tier | Mode | Verification |",
            "|---|---|---:|---|---|",
        ]
    )
    for row in accepted_outputs:
        if row["classification"] == "pre_existing_certificate":
            status = "Pre-existing certificate"
        elif row["is_new_solution"]:
            status = "**NEW**"
        else:
            status = "Calibration"
        lines.append(
            f"| {status} | [`{row['problem_id']}`]"
            f"(#{row['problem_id']}) | {row['tier']} | "
            f"{markdown_escape(row['evaluation_mode'])} | "
            f"{markdown_escape(row['verification_summary'])} |"
        )

    def append_solution_details(
        heading: str,
        selected: list[dict[str, Any]],
        *,
        show_submission: bool = True,
    ) -> None:
        lines.extend(["", heading, ""])
        for row in selected:
            if row["classification"] == "pre_existing_certificate":
                prefix = "PRE-EXISTING CERTIFICATE — "
            elif row["is_new_solution"]:
                prefix = "NEW — "
            else:
                prefix = ""
            lines.extend(
                [
                    f'<a id="{row["problem_id"]}"></a>',
                    f"### {prefix}`{row['problem_id']}`",
                    "",
                    row["mathematical_rendering"],
                    "",
                ]
            )
            if show_submission:
                lines.extend(
                    [
                        render_submitted_response(
                            row["submitted_response"]
                        ),
                        "",
                    ]
                )
            else:
                review = row["provenance_review"]
                certificate_path = review["certificate_path"]
                lines.extend(
                    [
                        review["audit_note"],
                        "",
                        (
                            "Fixed certificate: "
                            f"[`{certificate_path}`]"
                            f"(certificates/{row['problem_id']}.json)"
                        ),
                        "",
                    ]
                )
                if row["problem_id"] == "autocorr_signed_upper":
                    lines.extend(
                        [
                            (
                                "This verifies the one-sided signed "
                                "maximum stated above; it is not a bound "
                                "for the maximum absolute convolution."
                            ),
                            "",
                        ]
                    )

    append_solution_details(
        f"## New solutions — Tiers 1–3 ({len(new_solutions)})",
        new_solutions,
    )
    append_solution_details(
        (
            "## Pre-existing certificates — not scored "
            f"({len(external_certificates)})"
        ),
        external_certificates,
        show_submission=False,
    )
    append_solution_details(
        (
            "## Correct Tier 0 calibration solutions "
            f"({len(calibration_solutions)})"
        ),
        calibration_solutions,
    )

    lines.extend(
        [
            "## Numerically correct but rejected by Terra (16)",
            "",
            "| Problem | Tier | Numeric result | Terra |",
            "|---|---:|---|---:|",
        ]
    )
    for row in rejected_solutions:
        digits = row["numeric_accuracy"]["matching_digits"]
        lines.append(
            f"| [`{row['problem_id']}`](#rejected-{row['problem_id']}) | "
            f"{row['tier']} | Passed"
            f"{f' ({digits} matching digits)' if digits is not None else ''} | "
            f"{row['terra']['compliant_votes']}/"
            f"{row['terra']['total_valid_votes']} pass |"
        )

    for row in rejected_solutions:
        lines.extend(
            [
                "",
                f'<a id="rejected-{row["problem_id"]}"></a>',
                f"### REJECTED — `{row['problem_id']}`",
                "",
                row["mathematical_rendering"],
                "",
                render_submitted_response(row["submitted_response"]),
                "",
            ]
        )

    lines.extend(
        [
            "## Provenance",
            "",
            f"- Original result archive: `{archive.name}`",
            f"- Terra result file: `results/{TERRA_FILENAME}`",
            (
                "- Integrated rubric: "
                "`reports/compliance_checker_study/rubric_integrated.md`"
            ),
            (
                "- The JSON companion contains the complete submitted "
                "responses, per-trial Terra rationales, validator metrics, "
                "rendered mathematical forms, rejected candidates, and "
                "source metadata."
            ),
            "",
        ]
    )
    return artifact, "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
    )
    args = parser.parse_args()

    root = args.root.resolve()
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir
        else root / "reports/gpt56_pro_final_solutions"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    artifact, markdown = build_report(root, args.archive.resolve())
    (output_dir / "gpt56_pro_final_solutions.json").write_text(
        json.dumps(artifact, indent=2, ensure_ascii=False) + "\n"
    )
    (output_dir / "gpt56_pro_final_solutions.md").write_text(markdown)


if __name__ == "__main__":
    main()
