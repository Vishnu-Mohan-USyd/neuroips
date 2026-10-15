# methods draft

this draft describes the current two static regimes and the temporal model with uncapped, learned time constants.

we constructed a phenomenological firing-rate model comprising an orientation-selective sensory circuit and a recurrent sequence predictor. the task-prioritized and energy-prioritized static networks shared the same architecture and differed in their training objective. the sensory circuit incorporated SST-like inhibition, VIP-like disinhibition, and PV-like normalization. these roles were informed by experimental evidence for preferential VIP-to-SST inhibition, SST-mediated contextual suppression, and PV-mediated response-gain control. [Pfeffer et al., 2013](https://pubmed.ncbi.nlm.nih.gov/23817549/); [Adesnik et al., 2012](https://pmc.ncbi.nlm.nih.gov/articles/PMC3621107/); [Atallah et al., 2012](https://pmc.ncbi.nlm.nih.gov/articles/PMC3743079/). feedback modulation was represented through a nonlinear gain function, consistent at a qualitative level with evidence for nonlinear integration of cortical feedback. the particular gain function, Gaussian projections, and division into basal and prediction-driven SST components were modelling assumptions. [Fişek et al., 2023](https://pmc.ncbi.nlm.nih.gov/articles/PMC10244179/).

stimuli consisted of orientation angles on a circular domain with period $180^\circ$. the sensory representation contained 36 channels with preferred orientations $\theta_j=5j^\circ$, $j=0,\ldots,35$. a fixed encoder generated

$$
L_j(\theta)=
\exp\!\left[-\frac{d_{180}(\theta,\theta_j)^2}{2(12^\circ)^2}\right],
$$

where $d_{180}$ denotes the shortest circular angular distance. sensory drive was $D=1.6\,G_{1.1}L$, where $G_{1.1}$ was a nonnegative, row-normalized circular Gaussian projection with standard deviation 1.1 channels. channel preferences and feedforward weights remained fixed. training sequences contained 12 stimuli. initial orientation was sampled uniformly over the 36 channels, initial velocity uniformly over the integers $-4,\ldots,4$, and initial acceleration uniformly from $\{-1,0,1\}$. acceleration was retained with probability 0.9 and otherwise redrawn uniformly, including the possibility of drawing its previous value. latent velocity followed

$$
v_{n+1}=\operatorname{clip}(v_n+a_n,-4,4).
$$

orientations were obtained by integrating the applied velocities modulo 36. after the first transition, an applied velocity was replaced by zero with probability 0.02 when the preceding latent speed was at least two channels per transition. these single-transition halts altered stimulus displacement without altering the underlying velocity or acceleration recursion.

prediction was implemented by a conventional 64-unit tanh recurrent network followed by a learned affine projection to 36 next-orientation logits:

$$
h_n=\tanh(W_{\mathrm{in}}E_n+W_{\mathrm{rec}}h_{n-1}+b_h),
\qquad
\ell_n^{\mathrm{next}}=W_{\mathrm{fb}}h_n+b_{\mathrm{fb}}.
$$

the softmax of these logits supplied the incoming feedback for the following stimulus,

$$
f_{n+1}=\operatorname{softmax}(\ell_n^{\mathrm{next}}).
$$

the predictor received noiseless excitatory responses. hidden state and incoming feedback were initialized to zero at each sequence boundary. expected or unexpected condition labels were never supplied to the network. the static models performed one sensory-circuit evaluation and one recurrent update per stimulus.

the sensory circuit contained two 36-channel SST fields, nine VIP units, a nine-channel SST-like relay, and a scalar PV normalization signal. using column-vector notation and $[x]_+=\max(x,0)$, its intermediate activities were

$$
\begin{aligned}
b&=AD,\\
H&=[g_SAf-\theta_S]_+,\\
V&=[w_{vd}b-w_{vs}QH-\theta_V]_+,\\
V_{36}&=OV,\qquad p=K_{\mathrm{pred}}f,\\
S_B&=[w_{sd}Ob-\theta_S-w_{sv}V_{36}]_+,\\
S_P&=[g_SD\odot p-\theta_S-w_{sv}V_{36}]_+ .
\end{aligned}
$$

here $A$, $O$, and $Q$ respectively denote the $9\times36$ sensory-pooling, $36\times9$ output, and $9\times9$ relay-to-VIP projections. their nine population centers were spaced four orientation channels apart. the projections were normalized over their presynaptic dimension and used Gaussian standard deviations of $\sqrt2$, $2$, and $\sqrt2$ channels, respectively. the prediction-recruitment kernel $K_{\mathrm{pred}}$ was a **peak-normalized** circular Gaussian with standard deviation $\sqrt2$ channels. the final excitatory response was

$$
B=\frac{D}{1+mS_B},\qquad
q=B\odot\left[1+\tanh\!\left(g_Ef-mPS_P\right)\right],
$$

$$
PV=w_{\mathrm{PV}}\langle q\rangle_j,\qquad
E=\frac{q}{1+PV},
$$

where $P$ was a **row-normalized** circular Gaussian SST output projection with standard deviation two channels. fixed effective parameters were $m\approx0.2904474$, $\theta_S=\theta_V=0.02$, $w_{sd}=w_{sv}=w_{vd}=0.1$, $w_{vs}=0.5$, and $w_{\mathrm{PV}}=0.0025$. static excitatory feedback gain was $g_E\approx0.5457188$, and the initial SST gain was $g_S\approx0.2904474$. optional adaptation, additional local competition, and additional rate-saturation terms were disabled.

task performance combined current-orientation decoding and next-orientation prediction. current decoding used a fixed, confidence-weighted population-vector readout. independent Gaussian noise with standard deviation $\sigma=0.25A_{\mathrm{ref}}$ was added to each excitatory channel before rectification:

$$
a_j=[E_j+\eta_j]_+,\qquad \eta_j\sim\mathcal N(0,\sigma^2).
$$

with preferred orientations $\phi_j=j\pi/36$ expressed in radians, the doubled-angle resultant was

$$
X=\sum_j a_j\cos(2\phi_j),\qquad
Y=\sum_j a_j\sin(2\phi_j),\qquad
\rho=\sqrt{X^2+Y^2}.
$$

the 36 current-orientation logits were

$$
\ell_j^{\mathrm{current}}
=
8\,
\frac{\rho}{\rho+\sigma\sqrt{18}+\epsilon}
\frac{X\cos(2\phi_j)+Y\sin(2\phi_j)}{\rho+\epsilon},
\qquad
\epsilon=10^{-8}\,36R_{\mathrm{ref}}.
$$

confidence therefore scaled the logits before cross-entropy calculation. the decoder gain of eight remained fixed. $A_{\mathrm{ref}}$ was the median, across all 36 orientations, of the peak channel response in the initial circuit under zero feedback. decoder noise did not enter the sensory circuit or recurrent predictor. current cross-entropy, $C_{\mathrm{current}}$, averaged over all 12 stimulus positions; next-orientation cross-entropy, $C_{\mathrm{next}}$, averaged over the 11 adjacent stimulus transitions. their normalized combination was

$$
T=\frac{C_{\mathrm{current}}+C_{\mathrm{next}}}{2\ln36}.
$$

activity expenditure was represented by the fixed population-weighted proxy

$$
R=
\frac56\langle E\rangle
+\frac{37}{480}\langle PV\rangle
+\frac1{20}\left\langle\frac{S_B+S_P}{2}\right\rangle
+\frac{19}{480}\langle V\rangle.
$$

during static training, means included channels within each population, stimulus positions, and minibatch examples. SST expenditure used firing activity before multiplication by $m$ or application of the output projection. the weighting coefficients were model assumptions. activity penalties were motivated by energy-efficient coding considerations, while $R$ itself remained a dimensionless activity proxy. [Levy and Baxter, 1996](https://pubmed.ncbi.nlm.nih.gov/8868566/). the proxy omitted the recurrent predictor, auxiliary SST relay, and explicit synaptic and maintenance costs, which contribute to biological energy expenditure. [Hall et al., 2012](https://pmc.ncbi.nlm.nih.gov/articles/PMC3390246/). $R_{\mathrm{ref}}$ was measured before pretraining by presenting each orientation once with zero feedback and the local inhibitory circuit active. it remained fixed throughout training. the packaged reference values were $R_{\mathrm{ref}}\approx0.2200899$, $A_{\mathrm{ref}}\approx1.4213641$, and $\sigma\approx0.3553410$.

the static training objective was

$$
\mathcal L_{\mathrm{static}}
=(1-\alpha)T+\alpha\frac{R}{R_{\mathrm{ref}}},
$$

with $\alpha=0.07$ for the task-prioritized regime and $\alpha=0.70$ for the energy-prioritized regime. each seed first received 12,000 task-only pretraining updates, optimizing the recurrent predictor and feedback projection. both regimes then started from the same pretrained state and received 12,000 further updates, additionally optimizing the nonnegative SST input gain $g_S$. excitatory feedback gain, SST output gain $m$, spatial projections, thresholds, and other local circuit gains remained fixed. training used backpropagation through the complete sequence and Adam with learning rate $10^{-3}$, minibatch size 128, $\beta_1=0.9$, $\beta_2=0.999$, optimizer $\epsilon=10^{-8}$, zero weight decay, and global gradient-norm clipping at five. each arm used a fresh optimizer; seed-matched arms used matched stimulus and decoder-noise streams. the objective contained no response-shape target.

the temporal model extended the circuit with separate first-order states for effective excitatory feedback activation, $x$, and prediction-driven SST firing, $z$. distinguishing these effective kinetics is compatible with experimentally observed delays in SST visual responses, but the model's time constants were phenomenological parameters rather than estimates of cellular membrane constants. [Ma et al., 2010](https://pubmed.ncbi.nlm.nih.gov/20980594/). each stimulus was held for four relative time units, during which sensory drive and incoming prediction remained constant. pathway targets were

$$
x_\infty=g_Ef,\qquad
z_\infty=g_S[D\odot p-\theta_S-w_{sv}V_{36}]_+.
$$

thus, in the temporal model, SST gain multiplied the **rectified recruitment response**; in the static model it multiplied sensory–prediction coincidence before thresholding. its input-gain role in the auxiliary relay remained unchanged. the temporal states were evaluated analytically:

$$
x(t)=x(0)e^{-t/\tau_E}+x_\infty(1-e^{-t/\tau_E}),
$$

$$
z(t)=z(0)e^{-t/\tau_S}+z_\infty(1-e^{-t/\tau_S}).
$$

the excitatory response retained the preceding algebraic equations, replacing $g_Ef$ by $x(t)$ and $S_P$ by $z(t)$. basal SST and VIP remained instantaneous functions of the held inputs. responses were sampled at $t=0,0.1,\ldots,4.0$. during the subsequent five-unit blank, both pathway targets were zero, giving

$$
x_{\mathrm{next}}(0)=x(4)e^{-5/\tau_E},\qquad
z_{\mathrm{next}}(0)=z(4)e^{-5/\tau_S}.
$$

recurrent memory was retained during blanks. all states were reset at independent-sequence boundaries. time units had no assigned conversion to milliseconds.

temporal models inherited the seed-matched **common static pretraining state**. time constants were independently parameterized as $\tau_E=\operatorname{softplus}(u_E)$ and $\tau_S=\operatorname{softplus}(u_S)$, initialized at $(0.5,1)$ or $(1/3,1)$. these define the 2× and 3× initializations; their ratio and ordering were subsequently unrestricted. SST response gain also used a positive softplus parameterization. temporal fitting optimized the predictor, feedback projection, excitatory feedback gain, SST response gain, and both time constants. other circuit parameters remained fixed. fitting comprised 24,000 Adam updates at learning rate 0.003, with the same minibatch size, optimizer coefficients, and gradient clipping used for static training.

one population snapshot per stimulus was selected for decoding and recurrent prediction. selection maximized the squared magnitude of the noiseless orientation resultant over all 40 positive-time samples:

$$
t^*=\underset{t\in\{0.1,0.2,\ldots,4.0\}}{\arg\max}
\left|
\sum_j E_j(t)e^{2i\phi_j}
\right|^2.
$$

the procedure selected an entire population response at one time and did not use the true orientation label. exact ties selected the first candidate. Gaussian decoder noise was added only after selection, while the same selected noiseless response supplied one recurrent update for the next stimulus. gradients propagated through the selected response, without differentiating the discrete selection index. this readout examined the completed presentation window and therefore constituted an offline selection procedure. the objective rewarded the selected snapshot without an explicit early-time accuracy weight.

temporal activity expenditure included the presentation and its following blank, including the terminal blank of each sequence. for stimulus $n$, expenditure was

$$
J_n=
\operatorname{trapz}_{\,t=0,0.1,\ldots,4}R_n(t)
+
\frac1{40}\langle z_n(4)\rangle_j
\tau_S\left(1-e^{-5/\tau_S}\right).
$$

the second term analytically integrated residual prediction-SST firing during the blank; the factor $1/40$ combines the SST population weight with equal weighting of the basal and prediction-driven SST components. other accounted firing populations were zero during the blank. $J$ was the mean of $J_n$ across stimuli and minibatch examples. the activity penalty used a uniform weight per unit time and the fixed normalization $J_{\mathrm{ref}}=4R_{\mathrm{ref}}$. temporal training minimized

$$
\mathcal L_{\mathrm{temporal}}
=
0.7T+0.3\frac{J}{4R_{\mathrm{ref}}}
+\lambda(C_{\mathrm{current}}-c).
$$

the mean-current-cross-entropy ceiling $c$ was calibrated separately for each seed using a frozen temporal reference constructed from the common pretrained weights, with SST response-gain parameterization, peak selection, and both time constants set to one. calibration used 1,024 newly generated twelve-stimulus sequences, arranged in eight batches of 128, with stimulus and noise seeds 920001 and 920002. the multiplier started at zero and was updated after each optimizer step:

$$
\lambda_{k+1}
=
\max\!\left[0,\lambda_k+
\operatorname{stopgrad}(C_{\mathrm{current},k}-c)\right],
$$

using that step's pre-update minibatch cross-entropy and a multiplier step size of one.

expectation-dependent responses were evaluated using 216 matched pairs of five-stimulus histories. for every final orientation channel $y$ and velocity $v\in\{-3,-2,-1,1,2,3\}$, the expected history was

$$
[y-4v,\ y-3v,\ y-2v,\ y-v,\ y],
$$

and the unexpected history was

$$
[y+2v,\ y+v,\ y,\ y-v,\ y],
$$

with all channels taken modulo 36. these conditions shared the final stimulus and final physical transition; their earlier histories established continuation versus a reversal at the last transition. responses were aligned by channel preference relative to the presented orientation. the first response after state reset, aligned to its own stimulus orientation, provided the sensory baseline and retained local inhibition. the resulting profiles described activity across population channels. preferred-channel activity was measured at $0^\circ$, and broad-flank activity averaged the eight channels at $\pm15^\circ,\pm20^\circ,\pm25^\circ,\pm30^\circ$. responses were averaged over histories and the relevant channel group before division by the correspondingly averaged baseline. temporal early and late measurements additionally averaged over $t=0.1,0.2$ and $t=3.1,\ldots,4.0$, respectively. raw profiles and baseline-normalized measurements were retained separately.

static decoding for the paper figures used an evaluation-only cosine nearest-centroid classifier fitted to balanced expected and unexpected responses from each trained checkpoint. noisy responses were rectified, normalized by their L1 norm, and then normalized by their L2 norm. orientation centroids were calculated by averaging the normalized training responses within each orientation class and then L2-normalizing each centroid. classification selected the centroid with greatest cosine similarity. three cross-validation folds each held out one absolute velocity—5°, 10°, or 15° per transition—including both velocity signs and both expectation conditions. all 36 orientation classes remained represented during fitting. fitting used 32 noise repetitions with seed 910001; testing used 128 repetitions with seed 940001. expected and unexpected responses shared the same noise realization within each paired comparison. pooled across folds, each checkpoint contributed 27,648 test trials per condition. this classifier was fitted without changing network weights, and condition identity was not an input. separately retained current-cross-entropy measurements used the training population-vector readout.

temporal selected-snapshot accuracy, instantaneous accuracy, next-stimulus prediction, and integrated activity were evaluated on 1,024 newly generated twelve-stimulus sequences using the fixed population-vector readout. evaluation used stimulus seed 930001 and selected-snapshot noise seed 930002; instantaneous timecourse evaluation additionally used noise seed 930003 for samples outside the early reporting window. condition-specific decoding on the matched expectation probes used 32 noise repetitions, with a common noise table across conditions and time points. ceiling feasibility was assessed using **selected-snapshot** current cross-entropy with numerical tolerance $10^{-6}$. each static regime and temporal initialization used training seeds 8, 9, and 10. analyses used fixed-step final checkpoints, and plotted variability represented the minimum–maximum range across the three training seeds.

two qualifications matter when incorporating this text into the paper. **the temporal endpoints slightly exceeded their calibrated cross-entropy ceilings**, by approximately $0.0008$; the method should therefore describe the dual constraint procedure without claiming exact constraint satisfaction. also, **the recent mechanism figure uses older v8 networks at $\alpha=0.05$ and $0.20$**. those networks omitted the SST output projection $P$, so that figure requires a separate methods description from the current static v9 networks described above.
