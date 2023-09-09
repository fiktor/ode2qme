"""MCMC simulations of GinOE-stable.

Here GinOE-stable refers to the GinOE (Ginibre Orthogonal Ensemble)
conditioned on the real part of all eigenvalues being positive.
"""

import math
import numpy as np
import scipy.special

def _f4_large_x(x):
    """
    Compute the function $f_4(x)$ for $x >= 5$.

    The function is given by:
    $f_4(x) = -x^2/2 -\frac12 \ln(\pi/2) - \ln(erfc(x / \sqrt{2}))$
    """
    xm2 = x**(-2)
    numerator = (
        xm2 * (0.999999999998222323 +
        xm2 * (31.6591433701505694 +
        xm2 * (273.490498082387895 +
               596.921392509356213 * xm2))))
    denominator = (1 +
        xm2 * (34.1591433666909506 +
        xm2 * (346.555025170910110 +
        xm2 * (1130.26231438556648 +
        xm2 * (749.903497964347952 -
        169.399064806894677 * xm2)))))
    return numerator / denominator

def _f3_large_x(x):
    """
    Compute the function $f_3(x) = -\ln(erfc(x / \sqrt{2}))$ for $x >= 5$.
    """
    return x**2/2 + np.log(x * (math.pi/2)**0.5) + _f4_large_x(x)

def f3(x):
    """
    Compute $f_3(x)$ for $x >= 0$.
    """
    x = np.array(x)
    output = np.empty_like(x, dtype=np.float64)
    mask1 = x < 5
    output[mask1] = -np.log(scipy.special.erfc(x[mask1] / np.sqrt(2)))
    mask2 = ~mask1
    output[mask2] = _f3_large_x(x[mask2])
    return output

def _f3i_large_x(x3):
    """
    Compute $f_3^{-1}(x_3)$ for $x_3 >= 14.3$.
    """
    x3r0 = x3 - 0.5 * np.log(math.pi/2)
    x = np.sqrt(2 * x3r0)
    x3r = x3r0 - np.log(x)
    x = np.sqrt(2 * x3r)
    for _ in range(9):
        x3r = x3r0 - np.log(x) - _f4_large_x(x)
        x = np.sqrt(2 * x3r)
    return x

def f3i(x3):
    """
    Compute $f_3^{-1}(x_3)$ for $x_3 >= 0$.

    Note: $f_3^{-1}(x_3) = \sqrt{2} InverseErfc(exp(-x_3))$.
    """
    x3 = np.array(x3)
    output = np.empty_like(x3, dtype=np.float64)
    mask1 = x3 < 14.3
    output[mask1] = np.sqrt(2) * scipy.special.erfcinv(np.exp(-x3[mask1]))
    mask2 = ~mask1
    output[mask2] = _f3i_large_x(x3[mask2])
    return output

def ginoe_ypotential(ys):
    """
    Compute the function $-2 y^2 - ln(erfc(\sqrt{2} y))$ for y >= 0.

    We use direct formula (using scipy) for y <= 5 and a 2-point Pade approximant for y > 5.
    """
    ys = np.array(ys)
    # Initialize output array with the same shape as ys
    output = np.empty_like(ys, dtype=np.float64)

    # For y > 5 (the absolute error is under $2 * 10^{-13}$ for y > 4):
    mask1 = ys > 5
    ys1 = ys[mask1]
    y2 = ys1**2
    y4 = y2 * y2
    y6 = y4 * y2
    y8 = y4 * y4

    numerator = (-0.8904519590478536430137744382884281115310 -
                 9.869261681191832701731724404616295971553/y6 -
                 16.90359050157721106889984300899829825369/y4 -
                 7.423694049765192486628066963007045221308/y2)

    denominator = (1.000000000000000000000000000000000000000 +
                   2.174258030977644555812272331973294792748/y8 +
                   15.74732873975124317572963943709831902318/y6 +
                   21.22717377517984141370472593958542366059/y4 +
                   8.617751886323607565086681753902613859088/y2)

    result1 = 0.8904519590478536430137744382884281115310 + numerator / denominator - \
              np.log(0.3989422804014326779399460599343818684759 / ys1)

    output[mask1] = result1

    # For y <= 5 (should work for y <= 18)
    mask2 = ~mask1
    ys2 = ys[mask2]
    result2 = -2 * ys2**2 - np.log(scipy.special.erfc(np.sqrt(2) * ys2))

    output[mask2] = result2

    return output

def ginoe_ypotential_scalar(y):
    """
    Same as ginoe_ypotential, but for a single float y.
    """
    if y > 5:
        y2 = y**2
        y4 = y2 * y2
        y6 = y4 * y2
        y8 = y4 * y4

        numerator = (-0.8904519590478536430137744382884281115310 -
                     9.869261681191832701731724404616295971553/y6 -
                     16.90359050157721106889984300899829825369/y4 -
                     7.423694049765192486628066963007045221308/y2)

        denominator = (1.000000000000000000000000000000000000000 +
                       2.174258030977644555812272331973294792748/y8 +
                       15.74732873975124317572963943709831902318/y6 +
                       21.22717377517984141370472593958542366059/y4 +
                       8.617751886323607565086681753902613859088/y2)

        return (0.8904519590478536430137744382884281115310
            + numerator / denominator
            - np.log(0.3989422804014326779399460599343818684759 / y))

    return -2 * y**2 - np.log(scipy.special.erfc(np.sqrt(2) * y))

class GinOEMCMCState:
    def __init__(self, J, real_evals=None, complex_evals=None, pairs=None,
                 pos=None):
        """
        State of the MCMC simulation of GinOE

        Args:
            J (int): size of the GinOE matrix.
            real_evals: np.ndarray of real eigenvalues.
            complex_evals: np.ndarray of complex eigenvalues with positive
              imaginary part.
            pairs (list): list of length $\ceil{J/2}$ describing the pairs of
              eigenvalues to be processed in the current epoch. Each element is
              either:
                ('r', i): describes (real_evals[i],),
                ('R', i, j): describes (real_evals[i], real_evals[j]),
                ('C', i): describes (complex_evals[i], np.conj(complex_evals[i])).
            pos (int): index of the pair to be processed in the current move.
        """
        self.J = J
        self.real_evals = real_evals
        self.complex_evals = complex_evals
        self.pairs = pairs
        self.pos = pos
        self.real_lookup = None
        self.complex_lookup = None

    def check_initialized(self):
        """
        Check that the state is initialized correctly.
        """
        assert self.real_evals is not None
        assert self.complex_evals is not None
        assert len(self.real_evals) + 2 * len(self.complex_evals) == self.J
        assert len(self.pairs) == (self.J + 1) // 2
        assert 0 <= self.pos
        assert self.pos <= len(self.pairs)

    def init_lookup(self):
        """
        Initialize the lookup tables.
        """
        self.real_lookup = [None] * len(self.real_evals)
        self.complex_lookup = [None] * len(self.complex_evals)
        for i, p in enumerate(self.pairs):
            if p[0] == 'R':
                self.real_lookup[p[1]] = i
                self.real_lookup[p[2]] = i
            elif p[0] == 'C':
                self.complex_lookup[p[1]] = i
            else:
                assert p[0] == 'r'
                self.real_lookup[p[1]] = i

    def pair_to_evals(self, pair):
        """
        Convert an element of self.pairs to a tuple describing
        the corresponding eigenvalues.

        Args:
            pair: either ('R', i, j) or ('C', i)

        Returns:
            either ('R', real_evals[i], real_evals[j]) or
            ('C', *complex_evals[i])
        """
        assert pair[0] in ('R', 'C')
        if pair[0] == 'R':
            return ('R', self.real_evals[pair[1]], self.real_evals[pair[2]])
        else:
            return ('C', *self.complex_evals[pair[1]])

    def update_cur_pair(self, new_pair):
        """
        Update the current pair.

        Args:
            new_pair: either ('R', x1, x2) or ('C', x, y)
        """
        assert not self.at_end()
        pos = self.pos
        cur_pair = self.pairs[pos]
        if cur_pair[0] == 'R':
            if new_pair[0] == 'R':
                self.real_evals[cur_pair[1]] = new_pair[1]
                self.real_evals[cur_pair[2]] = new_pair[2]
            else:
                assert new_pair[0] == 'C'
                i1, i2 = sorted(cur_pair[1:])
                self.remove_r(i2)
                self.remove_r(i1)
                self.complex_evals.append((new_pair[1], new_pair[2]))
                self.complex_lookup.append(pos)
                self.pairs[pos] = ('C', len(self.complex_evals) - 1)
        else:
            assert cur_pair[0] == 'C'
            if new_pair[0] == 'C':
                self.complex_evals[cur_pair[1]] = new_pair[1:]
            else:
                assert new_pair[0] == 'R'
                self.remove_c(cur_pair[1])
                k = len(self.real_evals)
                self.real_evals.append(new_pair[1])
                self.real_lookup.append(pos)
                self.real_evals.append(new_pair[2])
                self.real_lookup.append(pos)
                self.pairs[pos] = ('R', k, k + 1)

    def remove_r(self, i):
        """
        Remove self.real_evals[i]
        """
        j = len(self.real_evals) - 1
        if i != j:
            self.real_evals[i] = self.real_evals[j]
            pair_pos = self.real_lookup[j]
            self.real_lookup[i] = pair_pos
            old_pair = self.pairs[pair_pos]
            if old_pair[0] == 'R':
                _, i1, i2 = old_pair
                if i1 == j:
                    i1 = i
                else:
                    assert i2 == j
                    i2 = i
                self.pairs[pair_pos] = ('R', i1, i2)
            else:
                assert old_pair[0] == 'r'
                self.pairs[pair_pos] = ('r', i)
        self.real_evals.pop()
        self.real_lookup.pop()

    def remove_c(self, i):
        """
        Remove self.complex_evals[i]
        """
        j = len(self.complex_evals) - 1
        if i != j:
            self.complex_evals[i] = self.complex_evals[j]
            pair_pos = self.complex_lookup[j]
            self.complex_lookup[i] = pair_pos
            old_pair = self.pairs[pair_pos]
            assert old_pair == ('C', j)
            self.pairs[pair_pos] = ('C', i)
        self.complex_evals.pop()
        self.complex_lookup.pop()

    def at_end(self):
        """
        Check if the current epoch is finished.
        """
        return self.pos >= len(self.pairs)

    @property
    def k(self):
        return len(self.real_evals)

    def new_epoch(self, rng):
        """
        Start a new epoch by randomly picking the pairs.

        This function modifies the state in place and reorders eigenvalues.
        """
        rng.shuffle(self.real_evals)
        rng.shuffle(self.complex_evals)
        k = self.k
        pairs_R = [('R', 2 * i, 2 * i + 1) for i in range(k // 2)]
        pairs_r = [('r', k - 1)] if k % 2 == 1 else []
        pairs_C = [('C', i) for i in range(len(self.complex_evals))]
        pairs = pairs_R + pairs_r + pairs_C
        assert len(pairs) == (self.J + 1) // 2
        rng.shuffle(pairs)
        self.pairs = pairs
        self.pos = 0
        self.init_lookup()
        self.check_initialized()

    def cur_pair(self):
        """
        Get the current pair.
        """
        return self.pairs[self.pos]

class ContinuousProposalDistribution:
    def get_pdf(self, *args, **kwargs):
        return self.get_density(*args, **kwargs) / self.total_mass

class GinOEStableRealProposal(ContinuousProposalDistribution):
    """
    Proposal distribution for real eigenvalues of GinOE-stable.
    """
    def __init__(self, J, bulk_density=None, soft_xmax=None, zero_charge=None):
        """
        Args:
            J (int): size of the GinOE matrix.
            bulk_density (float): density of eigenvalues in the bulk
                (i.e. for 1 << x << J**0.5).
            soft_xmax (float): soft upper bound for the eigenvalues;
                density decays as C * exp(-(x-soft_xmax)) for x > soft_xmax.
            zero_charge (float): total density bump near x = 0.
        """
        self.J = J
        if bulk_density is None:
            bulk_density = 1 / math.pi
        self.bulk_density = bulk_density
        if soft_xmax is None:
            soft_xmax = 1.06 * J**0.5
        self.soft_xmax = soft_xmax
        if zero_charge is None:
            zero_charge = 2 / math.pi
        self.zero_charge = zero_charge
        self.total_mass = bulk_density * (soft_xmax + 1) + zero_charge

        region_probs = np.array([
            self.zero_charge, self.bulk_density * self.soft_xmax,
            self.bulk_density])
        self.region_probs = region_probs / region_probs.sum()

    def get_density(self, x):
        """
        Get the density of the proposal distribution at x.

        Args:
            x (float): the real eigenvalue.
        """
        if x < 0:
            return 0
        return (
            self.zero_charge * math.exp(-x)
            + self.bulk_density * math.exp(-max(0, x - self.soft_xmax)))

    def sample(self, rng):
        """
        Sample from the proposal distribution.
        """
        # Algorithm is as follows:
        # 1. Choose a region to sample from out of the following options:
        #   (a) Near 0 (mass: zero_charge)
        #   (b) Bulk (mass: bulk_density * soft_xmax)
        #   (c) Tail (mass: bulk_density).
        # 2. Sample from the chosen region.
        region = rng.choice(3, p=self.region_probs)
        if region == 0:
            return rng.exponential()
        elif region == 1:
            return rng.uniform(0, self.soft_xmax)
        else:
            return self.soft_xmax + rng.exponential()

class GinOEStableComplexBulkProposal(ContinuousProposalDistribution):
    """
    Proposal distribution for sampling the complex eigenvalues from the bulk
    (i.e. excluding the additional charge near x = 0).
    """
    def __init__(self, J, density=None, soft_xmax=None, inner_ymax=None,
                 scale=None, offset=None):
        """
        Args:
            J (int): size of the GinOE matrix.
            density (float): 2 times the density of eigenvalues in the bulk
                (i.e. for 1 << x << J**0.5, 1 << y << J**0.5).
                Note: factor of 2 is used because we are only sampling $y > 0$
                and not conjugates of such eigenvalues in the lower half plane.
            soft_xmax (float): soft upper bound for real part of the eigenvalues;
            inner_ymax (float): upper bound for imaginary part of the eigenvalues
                of the bulk (inner ellipse);
            scale (float): ratio of the x axis and y axis of the ellipse
                used to fit the bulk region;
            offset (float): -x of the center of the ellipse used.

        Note:
            There are 2 ellipses discussed here:
            - Inner ellipse: the ellipse containing the expected bulk region.
            - Proposal ellipse: stretched version which is used to sample from:
                * total_mass is the mass of the proposal ellipse.
                * The distribution contains exponential decay near the
                  boundary of the proposal ellipse.
        """
        self.J = J
        if density is None:
            density = 2 / math.pi
        self.density = density
        if soft_xmax is None:
            soft_xmax = 1.06 * J**0.5
        self.soft_xmax = soft_xmax
        if inner_ymax is None:
            inner_ymax = 1.17 * J**0.5
        self.inner_ymax = inner_ymax
        if scale is None:
            scale = 1.22
        self.scale = scale
        if offset is None:
            offset = ((scale**2 * inner_ymax**2 - soft_xmax**2) / (2 * soft_xmax))
            assert offset > 0
        self.offset = offset
        # Add 1 to account for soft boundary:
        self.semiaxis_x = soft_xmax + offset + 1
        assert self.semiaxis_x > offset
        proposal_ymax = (self.semiaxis_x**2 - offset**2)**0.5 / scale
        self.proposal_ymax = proposal_ymax

        # To compute total_mass we scale the ellips to unit circle.
        sin_theta0 = offset / self.semiaxis_x
        cos_theta0 = (1 - sin_theta0**2)**0.5
        self.theta0 = math.asin(sin_theta0)
        area_on_unit_circle = (
                (math.pi / 2 - self.theta0) / 2 - sin_theta0 * cos_theta0 / 2)
        self.total_mass = (
                density * self.semiaxis_x**2 / scale * area_on_unit_circle)

    def get_xmax(self, y):
        if y < 0 or y > self.proposal_ymax:
            return 0
        res = (self.semiaxis_x**2 - (self.scale * y)**2)**0.5 - self.offset
        assert res >= 0
        return res

    def get_density(self, x, y):
        """
        Get the density of the proposal distribution at (x, y).

        Args:
            x (float): the real part of the eigenvalue.
            y (float): the imaginary part of the eigenvalue.
        """
        if x < 0 or y < 0 or y > self.proposal_ymax:
            return 0
        cur_xmax = self.get_xmax(y)
        if x < cur_xmax - 1:
            return self.density
        tail_mass = min(1, cur_xmax) * self.density
        tail_start_x = max(cur_xmax - 1, 0)
        assert x >= tail_start_x
        return tail_mass * math.exp(-x + tail_start_x)

    def sample(self, rng):
        """
        Sample from the proposal distribution.
        """
        # Algorithm is as follows:
        # 1. Sample (x, y) from the proposal ellipse:
        # 1.1. Sample r^2 from [0, 1] uniformly.
        # 1.2. Sample theta from [theta0, pi / 2] uniformly.
        # 1.3. Compute (x, y) by converting (r2**0.5, theta) from polar
        # coordinates.
        # 1.4. If x < 0, retry from step 1.1.
        # 2. If x > xmax(y) - 1, adjust x according to the tail distribution.

        x = -1
        while x < 0:
            r = rng.uniform()**0.5
            theta = rng.uniform(self.theta0, math.pi / 2)
            x = r * math.sin(theta) * self.semiaxis_x - self.offset
        y = r * math.cos(theta) * self.semiaxis_x / self.scale
        cur_xmax = self.get_xmax(y)
        if x <= cur_xmax - 1:
            return x, y
        tail_start_x = max(cur_xmax - 1, 0)
        dx = x - tail_start_x
        assert dx > 0
        dx_max = cur_xmax - tail_start_x
        assert dx_max > dx
        x = tail_start_x - math.log(1 - dx / dx_max)
        return x, y

class GinOEStableComplexLocalProposal(ContinuousProposalDistribution):
    """
    Proposal distribution for sampling the complex eigenvalues from
    the neighbourhood of x = 0.
    """
    def __init__(self, J, bulk_density=None, soft_ymax=None, zero_charge=None):
        """
        Args:
            J (int): size of the GinOE matrix.
            bulk_density (float): density of eigenvalues in the bulk
                (i.e. for 1 << y << J**0.5).
            soft_ymax (float): soft upper bound for the eigenvalues;
                density decays as C * exp(-(y-soft_ymax)) for y > soft_ymax.
            zero_charge (float): total density bump near y = 0.
        """
        if bulk_density is None:
            bulk_density = 1.4 * J**0.5 / math.pi
        if soft_ymax is None:
            soft_ymax = 1.25 * J**0.5
        if zero_charge is None:
            zero_charge = 1
        # We use the same model as for GinOEStableRealProposal
        # for sampling y. x is sampled from the exponential distribution.
        self._model = GinOEStableRealProposal(
                J, bulk_density, soft_ymax, zero_charge)

    @property
    def total_mass(self):
        return self._model.total_mass

    def get_density(self, x, y):
        """
        Get the density of the proposal distribution at (x, y).

        Args:
            x (float): the real part of the eigenvalue.
            y (float): the imaginary part of the eigenvalue.
        """
        if x < 0 or y < 0:
            return 0
        return self._model.get_density(y) * math.exp(-x)

    def sample(self, rng):
        """
        Sample from the proposal distribution.
        """
        y = self._model.sample(rng)
        x = rng.exponential()
        return x, y

class GinOEStableComplexProposal(ContinuousProposalDistribution):
    """
    Proposal distribution for sampling the complex eigenvalues.

    Combines GinOEStableComplexBulkProposal and GinOEStableComplexLocalProposal.
    """
    def __init__(self, J, bulk_args=None, local_args=None):
        self.J = J
        bulk_args = bulk_args or {}
        local_args = local_args or {}
        self._bulk_model = GinOEStableComplexBulkProposal(J, **bulk_args)
        self._local_model = GinOEStableComplexLocalProposal(J, **local_args)
        self.total_mass = (
                self._bulk_model.total_mass + self._local_model.total_mass)

    def get_density(self, x, y):
        """
        Get the density of the proposal distribution at (x, y).

        Args:
            x (float): the real part of the eigenvalue.
            y (float): the imaginary part of the eigenvalue.
        """
        bulk_density = self._bulk_model.get_density(x, y)
        local_density = self._local_model.get_density(x,y)
        return bulk_density + local_density

    def sample(self, rng):
        """
        Sample from the proposal distribution.
        """
        if rng.uniform() < self._bulk_model.total_mass / self.total_mass:
            return self._bulk_model.sample(rng)
        else:
            return self._local_model.sample(rng)

class GinOEStablePairProposal:
    """
    Proposal distribution for sampling pairs of eigenvalues.

    Its sample method returns either:
        (a) tuple ('R', x0, x1): two real eigenvalues;
        (b) tuple ('C', x, y) representing a pair x \pm i y
            of complex eigenvalues.
    """
    def __init__(self, J, p_complex=None, real_args=None, complex_args=None):
        assert J >= 2
        self.J = J
        if isinstance(real_args, GinOEStableRealProposal):
            assert real_args.J == J
            self._real_model = real_args
        else:
            real_args = real_args or {}
            self._real_model = GinOEStableRealProposal(J, **real_args)

        if isinstance(complex_args, GinOEStableComplexProposal):
            assert complex_args.J == J
            self._complex_model = complex_args
        else:
            complex_args = complex_args or {}
            self._complex_model = GinOEStableComplexProposal(J, **complex_args)

        if p_complex is None:
            # Note: parity_correction is J / (2 * (J // 2))
            # We ignore it here, in the proposal distribution,
            # since it will be taken into account in the acceptance ratio.
            # That may lower the acceptance ratio slightly, but for
            # large J (which are the most computationally expensive)
            # the AR is multiplied by (1 - 1 / J) or something even closer to 1.
            parity_correction = 1
            complex_mass = self._complex_model.total_mass
            real_mass = self._real_model.total_mass
            p_complex = parity_correction * complex_mass / (complex_mass + real_mass)
            assert 0 <= p_complex
            assert p_complex <= 1
        self.p_complex = p_complex
        self.p_real = 1 - p_complex

    def get_pdf(self, pair_type, x0, x1):
        if pair_type == 'R':
            return (self.p_real * self._real_model.get_density(x0)
                    * self._real_model.get_density(x1))
        assert pair_type == 'C'
        return self.p_complex * self._complex_model.get_density(x0, x1)

    def sample(self, rng):
        if rng.uniform() < self.p_real:
            x0 = self._real_model.sample(rng)
            x1 = self._real_model.sample(rng)
            return 'R', x0, x1
        else:
            return 'C', *self._complex_model.sample(rng)

class GinOEStableMCMC:
    """
    Sampling eigenvalues from GinOE-stable using MCMC.
    """
    def __init__(self, J, seed=None):
        """
        Args:
            J (int): size of the GinOE matrix.
        """
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        assert J >= 2
        self.J = J
        self.state = GinOEMCMCState(J, [], [], [], 0)
        self.real_proposal = GinOEStableRealProposal(J)
        self.pair_proposal = GinOEStablePairProposal(
            J, real_args=self.real_proposal)
        self.init_evals()
        self.epoch_i = -1

    def init_evals(self):
        assert len(self.state.real_evals) == 0
        assert len(self.state.complex_evals) == 0
        if self.J % 2 == 1:
            self.state.real_evals.append(self.real_proposal.sample(self.rng))
        for i in range(self.J // 2):
            pair = self.pair_proposal.sample(self.rng)
            if pair[0] == 'R':
                self.state.real_evals.append(pair[1])
                self.state.real_evals.append(pair[2])
            else:
                assert pair[0] == 'C'
                self.state.complex_evals.append((pair[1], pair[2]))
        self.state.real_evals = self.state.real_evals
        self.state.complex_evals = self.state.complex_evals

    def run(self, n_epochs=1, callback=None, verbose=1):
        """
        Run the MCMC for n_epochs epochs.
        """
        stats = {
            "J": self.J,
            "seed": self.seed,
            "epoch_range": (self.epoch_i + 1, self.epoch_i + n_epochs + 1),
            "acceptance_rate": []
            }
        for epoch_i in range(n_epochs):
            self.state.new_epoch(self.rng)
            self.epoch_i += 1
            num_total = 0
            num_accepted = 0
            while not self.state.at_end():
                num_total += 1
                num_accepted += self.move(callback=callback)
            stats["acceptance_rate"].append(num_accepted / num_total)
            if verbose > 1:
                print(f"[{self.epoch_i}: {num_accepted/num_total:.3f}]", end="")
            elif verbose == 1 and epoch_i % 100 == 0:
                c = f"{epoch_i % 10000 // 1000}" if epoch_i % 1000 == 0 else "."
                print(c, end="")
        print("")
        return stats

    def move(self, callback=None):
        """
        Perform one MCMC move.
        """
        assert not self.state.at_end()
        cur_pair = self.state.cur_pair()
        if cur_pair[0] == 'r':
            res = self._move_real1(cur_pair[1])
        else:
            res = self._move_pair(cur_pair)
        self.state.pos += 1
        if callback is not None:
            callback(self.state, self.rng)
        return res

    def _move_real1(self, i):
        """
        Move a single real eigenvalue.
        """
        return self._move_real1_to(i, self.real_proposal.sample(self.rng))

    def _move_real1_to(self, i, x_new):
        """
        Move a single real eigenvalue.
        """
        x_old = self.state.real_evals[i]
        if self.rng.uniform() < self._acceptance_ratio_real(i, x_old, x_new):
            self.state.real_evals[i] = x_new
            return True
        else:
            return False

    def _move_pair(self, pair):
        """
        Move a pair of eigenvalues.
        """
        pair_old = self.state.pair_to_evals(pair)
        pair_new = self.pair_proposal.sample(self.rng)
        if pair_old[0] == 'R' and pair_new[0] == 'R':
            res1 = self._move_real1_to(pair[1], pair_new[1])
            res2 = self._move_real1_to(pair[2], pair_new[2])
            return (res1 + res2) / 2
        else:
            acceptance_ratio = self._acceptance_ratio_pair(pair, pair_old, pair_new)
            if self.rng.uniform() < acceptance_ratio:
                self.state.update_cur_pair(pair_new)
                return True
            else:
                return False

    @staticmethod
    def _np_array(arr, shape_type):
        """
        Converts a list with floats to np.ndarray.

        Args:
            arr (list): list of floats (for shape_type 'R')
                or pairs of floats (for shape_type 'C').
            shape_type: 'R' or 'C'. Note that this is needed in order to
                handle the case len(arr) == 0 correctly.
        """
        if len(arr) == 0 and shape_type == 'C':
            res = np.array(arr).reshape((0, 2))
        else:
            res = np.array(arr)
        assert res.dtype == np.float64, f"(dtype={res.dtype})"
        return res

    @staticmethod
    def _np_array_except(arr, shape_type, indices):
        """
        Converts a list with floats to np.ndarray with indices removed.

        Args:
            arr (list): list of floats (for shape_type 'R')
                or pairs of floats (for shape_type 'C').
            shape_type: 'R' or 'C'. Note that this is needed in order to
                handle the case len(arr) == 0 correctly.
            indices: indices to remove.
        """
        arr = GinOEStableMCMC._np_array(arr, shape_type)
        indices = sorted(indices)
        imax = len(arr) - 1
        for i in reversed(indices):
            arr[i] = arr[imax]
            imax -= 1
        return arr[:-len(indices)]

    def _acceptance_ratio_real(self, i, x_old, x_new):
        """
        Compute the acceptance ratio for a move of a single real eigenvalue.

        Args:
            i (int): index of the eigenvalue to move.
            x_old (float): old value of the eigenvalue.
            x_new (float): new value of the eigenvalue.
        """
        res = math.exp(x_old**2 / 2 - x_new**2 / 2)
        real_evals = self._np_array_except(self.state.real_evals, 'R', (i,))
        res *= np.abs(np.prod((x_new - real_evals) / (x_old - real_evals)))
        complex_evals = self._np_array(self.state.complex_evals, 'C')
        if len(complex_evals) == 0:
            complex_evals.reshape((0, 2))
        assert complex_evals.shape == (len(self.state.complex_evals), 2), (
            f"{complex_evals.shape} != {(len(self.state.complex_evals), 2)}")
        new_diffs = (x_new - complex_evals[:, 0])**2 + complex_evals[:, 1]**2
        old_diffs = (x_old - complex_evals[:, 0])**2 + complex_evals[:, 1]**2
        res *= np.abs(np.prod(new_diffs / old_diffs))
        res *= (self.real_proposal.get_pdf(x_old)
            / self.real_proposal.get_pdf(x_new))
        assert res > 0
        return res

    def _acceptance_ratio_pair(self, pair, pair_old, pair_new):
        """
        Compute the acceptance ratio for a move of a pair of eigenvalues.

        Args:
            pair (tuple): element of self.state.pairs describing a pair
                of eigenvalues to move.
            pair_old: tuple describing old values of the eigenvalues.
            pair_new: tuple describing new values of the eigenvalues.
        """
        if pair_old[0] == 'R':
            # 'R' -> 'R' is handled in _move_pair.
            assert pair_new[0] == 'C'
            return self._acceptance_ratio_pair_r2c(pair, pair_old, pair_new)
        else:
            assert pair_old[0] == 'C'
            if pair_new[0] == 'R':
                return 1 / self._acceptance_ratio_pair_r2c(
                        pair, pair_new, pair_old)
            else:
                assert pair_new[0] == 'C'
                return self._acceptance_ratio_pair_c2c(
                        pair[1], pair_old, pair_new)

    def _acceptance_ratio_pair_r2c(self, pair, pair_old, pair_new):
        if pair[0] == 'R':
            real_evals = self._np_array_except(self.state.real_evals, 'R', pair[1:])
            complex_evals = self._np_array(self.state.complex_evals, 'C')
            assert complex_evals.shape == (len(self.state.complex_evals), 2)
        else:
            assert pair[0] == 'C'
            real_evals = self._np_array(self.state.real_evals, 'R')
            complex_evals = self._np_array_except(
                    self.state.complex_evals, 'C', pair[1:])
            assert complex_evals.shape == (len(self.state.complex_evals) - 1, 2)
        assert pair_old[0] == 'R'
        assert pair_new[0] == 'C'
        x1, x2 = pair_old[1:]
        x, y = pair_new[1:]
        k = len(real_evals)
        res_const = (k + 2) * (k + 1) / (k // 2 + 1)
        res_pdf = (
            2 * self.pair_proposal.get_pdf('R', x1, x2)
            / self.pair_proposal.get_pdf('C', x, y))
        res_pe = ((x1**2 + x2**2) / 2
                - (x**2 + y**2) - ginoe_ypotential_scalar(y))
        res_self = math.log(abs(2 * y / (x2 - x1)))
        res_real = np.sum(np.log(np.abs(
            ((x - real_evals)**2 + y**2)
            / ((x1 - real_evals) * (x2 - real_evals)))))
        res_complex = np.sum(np.log(np.abs(
            ((x - complex_evals[:, 0])**2 + (y - complex_evals[:, 1])**2)
            * ((x - complex_evals[:, 0])**2 + (y + complex_evals[:, 1])**2)
            / (((x1 - complex_evals[:, 0])**2 + complex_evals[:, 1]**2)
                * ((x2 - complex_evals[:, 0])**2 + complex_evals[:, 1]**2)))))
        res = res_const * res_pdf * math.exp(res_pe + res_self + res_real + res_complex)
        assert res > 0, (
            f"res = {res} = {res_const:.3g} * {res_pdf:.3g} * exp("
            + f"{res_pe:.3g} + {res_self:.3g} + {res_real:.3g} "
            + f"+ {res_complex:.3g})")
        return res

    def _acceptance_ratio_pair_c2c(self, i, pair_old, pair_new):
        real_evals = self._np_array(self.state.real_evals, 'R')
        complex_evals = self._np_array_except(
                self.state.complex_evals, 'C', (i,))
        assert complex_evals.shape == (len(self.state.complex_evals) - 1, 2), (
                f"shape={complex_evals.shape}, dtype={complex_evals.dtype}")
        assert pair_old[0] == 'C'
        assert pair_new[0] == 'C'
        x_old, y_old = pair_old[1:]
        x_new, y_new = pair_new[1:]
        res_pe = ((x_old**2 + y_old**2) + ginoe_ypotential_scalar(y_old)
                - (x_new**2 + y_new**2) - ginoe_ypotential_scalar(y_new))
        res_self = math.log(y_new / y_old)
        res_real = np.sum(np.log(np.abs(
            ((x_new - real_evals)**2 + y_new**2)
            / ((x_old - real_evals)**2 + y_old**2))))
        dx_new2 = (x_new - complex_evals[:, 0])**2
        dx_old2 = (x_old - complex_evals[:, 0])**2
        res_complex_pos = np.sum(np.log(np.abs(
            (dx_new2 + (y_new - complex_evals[:, 1])**2)
            / (dx_old2 + (y_old - complex_evals[:, 1])**2))))
        res_complex_neg = np.sum(np.log(np.abs(
            (dx_new2 + (y_new + complex_evals[:, 1])**2)
            / (dx_old2 + (y_old + complex_evals[:, 1])**2))))
        res_pdf = (self.pair_proposal.get_pdf('C', x_old, y_old)
            / self.pair_proposal.get_pdf('C', x_new, y_new))
        res = res_pdf * math.exp(
            res_pe + res_self + res_real + res_complex_pos + res_complex_neg)
        assert res > 0, (
            f"res = {res} = {res_pdf:.3g} * exp({res_pe:.3g} + {res_self:.3g} "
            + f"+ {res_real:.3g} + {res_complex_pos:.3g} "
            + f"+ {res_complex_neg:.3g})")
        return res

class GSampler:
    """Sample matrix $G = - Q R Q^T$ given the eigenvalues of $R$."""

    @staticmethod
    def truncated_normal_sample(threshold, rng):
        """Sample from truncated normal distribution."""
        # Algorithm
        # 1. Compute x3min = f3[x0]
        # 2. Sample dx3 from exponential distribution
        # 3. Compute x3 = x3min + dx3
        # 4. Return x1 = f3i[x3], where f3i is the inverse function of f3.
        threshold = np.array(threshold, dtype=np.float64)
        x3min = f3(threshold)
        dx3 = rng.exponential(size=threshold.shape)
        x3 = x3min + dx3
        return f3i(x3)

    @staticmethod
    def sample_bc(ys, rng):
        """Samples $b$ and $c$ conditional on $y$.

        The sampling procedure is equivalent to the following:
        1. Sample $h = b + c$ from $N(0, 1)$ conditioned on $h > 2 * y$.
        2. Sample $s = \sign(b - c) = \pm 1$ with equal probability.
        3. Compute $\delta = b - c = s\sqrt{h^2 - 4 * y^2}$.
        4. Return $b = (h + \delta) / 2$ and $c = (h - \delta) / 2$.

        This function is designed to work for $0 <= ys <= 100$.

        Args:
            y = \sqrt{bc} --- np.ndarray of dtype float64.
            rng --- np.random.Generator.

        Returns: pair of np.ndarray's of the same length as y.
        """
        assert np.all(0 <= ys)
        assert np.all(ys <= 100.0), f"max(ys) = {np.max(ys)}"
        hs = GSampler.truncated_normal_sample(2 * ys, rng)
        ss = rng.choice([-1, 1], size=ys.shape)
        delta2s = hs**2 - 4 * ys**2
        assert np.all(delta2s > -5.0e-11), f"min(delta2s) = {np.min(delta2s)}"
        deltas = ss * np.sqrt(delta2s)
        return (0.5 * (hs + deltas), 0.5 * (hs - deltas))

    @staticmethod
    def _sample_R(real_evals, xs, bs, cs, rng):
        """
        Construct R from the block-diagonal data, sampling upper triangular part.
        """
        k = len(real_evals)
        k1 = len(xs)
        J = k + 2 * k1
        rmatrix = np.zeros((J, J), dtype=np.float64)

        idx1 = np.arange(k)
        idx2 = k + 2 * np.arange(k1)
        idx3 = idx2 + 1
        rmatrix[idx1, idx1] = real_evals
        rmatrix[idx2, idx2] = xs
        rmatrix[idx3, idx3] = xs
        rmatrix[idx2, idx3] = bs
        rmatrix[idx3, idx2] = -cs

        idx = np.arange(J)
        mask = idx[:, np.newaxis] < idx[np.newaxis, :]
        mask[idx2, idx3] = False
        mask_size = J * (J - 1) // 2 - k1
        rmatrix[mask] = rng.normal(size=mask_size)
        return rmatrix

    @staticmethod
    def sample_R(real_evals, complex_evals, rng):
        """
        Sample R given the eigenvalues.

        Args:
            real_evals: np.ndarray of length k & dtype np.float64.
            complex_evals: np.ndarray of length (k1, 2) and dtype np.float64,
                where k1 = (J - k) // 2.
            rng: np.random.Generator.

        Returns:
            np.ndarray of shape (J, J) and dtype np.float64 describing matrix R.
        """
        xs = complex_evals[:, 0]
        ys = complex_evals[:, 1]
        bs, cs = GSampler.sample_bc(ys, rng)
        return GSampler._sample_R(real_evals, xs, bs, cs, rng)

    @staticmethod
    def sample_orthogonal(J, rng):
        """
        Sample JxJ orthogonal matrix Q.
        """
        random_matrix = rng.normal(size=(J, J))
        Q, R = np.linalg.qr(random_matrix)
        return Q

    @staticmethod
    def sample_G(real_evals, complex_evals, rng):
        """
        Sample G = -Q R Q^T given the eigenvalues of R.
        """
        R = GSampler.sample_R(real_evals, complex_evals, rng)
        J = len(real_evals) + len(complex_evals) * 2
        Q = GSampler.sample_orthogonal(J, rng)
        return -Q @ R @ Q.T

    @staticmethod
    def sample_c(J, rng):
        """
        Sample c.

        Note that the normal distribution is scaled by 1/d**0.5 because
        c is multiplied by I instead of $F_{0} = I/\sqrt{d}$.
        """
        d = int((J+1)**0.5 + 0.5)
        assert J == d**2 - 1
        return rng.normal(size=J) / np.sqrt(d)

    @staticmethod
    def sample_Gc(real_evals, complex_evals, rng):
        """
        Sample (G, c).
        """
        real_evals = np.array(real_evals)
        complex_evals = np.array(complex_evals)
        J = len(real_evals) + len(complex_evals) * 2
        G = GSampler.sample_G(real_evals, complex_evals, rng)
        c = GSampler.sample_c(J, rng)
        return G, c

def gellmann_matrices(d):
    """
    Return the list of J dxd matrices as an np.ndarray.

    Here J = d^2 - 1. The matrices F_j are traceless matrices
    satisfying Tr(F_j @ F_k) = delta_{j,k}.

    Args:
        d: a positive integer.

    Returns: np.ndarray of dtype np.complex128 and shape (J, d, d).
    """
    J = d ** 2 - 1
    matrices = np.zeros((J, d, d), dtype=np.complex128)

    index = 0

    # Diagonal matrices
    for i in range(1, d):
        diag_elements = np.zeros(d)
        diag_elements[:i] = 1
        diag_elements[i] = -i
        matrices[index] = np.diag(diag_elements / np.sqrt(i * (i + 1)))
        index += 1

    # Off-diagonal real matrices
    for i in range(d):
        for j in range(i+1, d):
            mat = np.zeros((d, d))
            mat[i, j] = 1
            mat[j, i] = 1
            matrices[index] = mat / 2**0.5
            index += 1

    # Off-diagonal imaginary matrices
    for i in range(d):
        for j in range(i+1, d):
            mat = np.zeros((d, d), dtype=np.complex128)
            mat[i, j] = -1j
            mat[j, i] = 1j
            matrices[index] = mat / 2**0.5
            index += 1

    return matrices

class SparseGellMann:
    """
    Sparse description of dxd Gell-Mann matrices.

    Includes the following components:
        diag_matrices: a (d - 1) x d np.ndarray describing the diagonal
            matrices.
        real_pairs: a (d * (d-1) // 2) x 2 np.ndarray describing the
            real off-diagonal matrices. This is a list of all pairs
            (i, j) with i < j.
            Corresponding real matrix F has the following elements:
            - F[i, j] = 2**(-0.5)
            - F[j, i] = 2**(-0.5)
            Corresponding imaginary matrix F has the following elements:
            - F[i, j] = 2**(-0.5) * 1j
            - F[j, i] = 2**(-0.5) * (-1j)
    """
    def __init__(self, d):
        self.d = d
        self.J = d**2 - 1
        diag_matrices = np.zeros((d - 1, d), dtype=np.complex128)
        for i in range(d-1):
            ci = 1 / np.sqrt((i + 1) * (i + 2))
            diag_matrices[i, :(i+1)] = ci
            diag_matrices[i, i+1] = -ci * (i + 1)
        self.diag_matrices = diag_matrices
        self.real_pairs0 = np.array([
            i for i in range(d) for j in range(i+1, d)])
        self.real_pairs1 = np.array([
            j for i in range(d) for j in range(i+1, d)])
        self.range_d = np.arange(d)
        self.j_real_start = d - 1
        self.j_imag_start = (d - 1) * (d + 2) // 2
        self.real_ij = np.arange(self.j_real_start, self.j_imag_start)
        self.imag_ij = np.arange(self.j_imag_start, self.J)

    def times_F(self, x):
        """
        Convert tensor x into sum_j x[..., j] F_j.
        """
        res_shape = x.shape[:-1] + (self.d, self.d)
        assert x.shape == res_shape[:-2] + (self.J,)
        res = np.zeros(res_shape, dtype=np.complex128)
        res[..., self.range_d, self.range_d] = (
                x[..., :self.j_real_start] @ self.diag_matrices)
        res_real = x[..., self.j_real_start:self.j_imag_start] * 2**(-0.5)
        res_imag = x[..., self.j_imag_start:] * (1j * 2**(-0.5))
        res[..., self.real_pairs0, self.real_pairs1] = (
                res_real - res_imag)
        res[..., self.real_pairs1, self.real_pairs0] = (
                res_real + res_imag)
        return res

    def times_newF(self, x):
        """
        Compute x @ F_j for every j, and adds index j to the result.

        res[..., j, n] = (x @ F_j)[..., n].
        """
        res_shape = x.shape[:-1] + (self.J, self.d)
        assert x.shape == res_shape[:-2] + (self.d,)
        res = np.zeros(res_shape, dtype=np.complex128)
        res[..., :self.j_real_start, :] = (
            x[..., np.newaxis, :] * self.diag_matrices)
        x_real0 = x[..., self.real_pairs0] * 2**(-0.5)
        res[..., self.real_ij, self.real_pairs1] = x_real0
        res[..., self.imag_ij, self.real_pairs1] = (-1j) * x_real0
        x_imag0 = x[..., self.real_pairs1] * 2**(-0.5)
        res[..., self.real_ij, self.real_pairs0] = x_imag0
        res[..., self.imag_ij, self.real_pairs0] = (+1j) * x_imag0
        return res

    def gc_to_a(self, G, c):
        """
        Using the Liouvillian in ODE form, compute $a$ from QME form.

        Args:
            G: JxJ matrix with real elements,
            c: vector of length J with real elements.

        Returns:
            a: JxJ Hermitian matrix with complex elements.
        """
        # The formula is
        # $$
        # a_{m,n} = \sum_{i=1}^{J} \Tr\left(
        #       \left(\sum_{j=1}^{J} G_{i,j} \F_j + c_i I\right) F_m F_i F_n
        #   \right).
        # $$
        assert G.shape == (self.J, self.J)
        assert c.shape == (self.J,)
        # \sum_{j=1}^{J} G_{i,j} \F_j + c_i I:
        res = (
            self.times_F(G)
           + c[:, np.newaxis, np.newaxis] * np.eye(self.d)[np.newaxis, :, :])
        assert res.shape == (self.J, self.d, self.d) # (i, a, b)
        # * F_m:
        res = self.times_newF(res)
        assert res.shape == (self.J, self.d, self.J, self.d) # (i, a, m, c)
        # * F_i:
        res = np.moveaxis(res, 0, 3) # (a, m, c, i)
        res = self.times_F(res)
        assert res.shape == (self.d, self.J, self.d, self.d, self.d) # (a, m, c, c, d)
        # Contract extra indices:
        res = np.einsum('amccd->mad', res)
        # * F_n:
        res = self.times_newF(res)
        assert res.shape == (self.J, self.d, self.J, self.d) # (m, a, n, a)
        res = np.einsum('mana->mn', res)
        return res

    def gc_to_h(self, G, c):
        """
        Using the Liouvillian in ODE form, compute $H$ from QME form.

        Args:
            G: JxJ matrix with real elements,
            c: vector of length J with real elements.

        Returns:
            H: dxd Hermitian matrix with complex elements.
        """
        # The formula is:
        # $$ H = \sum_{m,n=1}^{J} G_{n,m} [F_m, F_n] / (2j * d).$$
        assert G.shape == (self.J, self.J)
        assert c.shape == (self.J,)
        res = self.times_F(np.moveaxis(self.times_F(G), 0, 2))
        res = np.einsum('abbc->ac', res) - np.einsum('bcab->ac', res)
        return res / (2j * self.d)

    def ah_to_g(self, a, H):
        """
        Using the Liouvillian in QME form, compute $G$ from ODE form.

        Args:
            a: Hermitian JxJ matrix with complex elements,
            H: Hermitian dxd matrix with complex elements.

        Returns:
            G: JxJ matrix with real elements.
        """
        # The formula is
        # $$
        # G_{ij} = \Tr(F_i \mcL(F_j))
        #   = -1j * \Tr(F_i [H, F_j]) + \sum_{m,n=1}^{J} \Tr(F_i a_{m,n} (
        #       F_m F_j F_n - {F_n F_m, F_j}/2)).
        # $$
        # Commutator in the first term can be replaced with 2*Re.
        assert a.shape == (self.J, self.J)
        assert H.shape == (self.d, self.d)
        res_h = self.times_newF(self.times_newF(H))
        res_h = 2j * np.einsum('nijn->ij', res_h)

        # a_{m,n} F_m F_j F_n:
        res_a1 = np.moveaxis(self.times_newF(self.times_F(a.T)), 0, 3)
        res_a1 = np.einsum('ajbbc->ajc', self.times_F(res_a1))

        # a_{m,n} F_n F_m:
        res_a2 = self.times_F(np.moveaxis(self.times_F(a), 0, 2))
        res_a2 = np.einsum('abbc->ac', res_a2)
        # {..., F_j} / 2
        res_a2 = (self.times_newF(res_a2) + self.times_newF(res_a2.T.conj()).T.conj()) / 2

        res_a = res_a1 - res_a2
        res_a = np.einsum('ajia->ij', self.times_newF(res_a))
        return res_h + res_a

def wilson_interval(num_trials, num_successes):
    """
    Computes Wilson score interval with continuity correction.

    Implementation follows
    https://en.wikipedia.org/w/index.php?title=Binomial_proportion_confidence_interval&oldid=1172639005#Wilson_score_interval_with_continuity_correction
    """
    p_hat = num_successes / num_trials
    n = num_trials
    alpha = 0.05
    z = 1.96 # 95% confidence interval
    denominator = 2 * n + z**2
    mu = 2 * n * p_hat + z**2
    var_mean = 4 * n * p_hat * (1 - p_hat) + z**2 - 1 / n
    var_delta = 4 * p_hat - 2
    # These are always between 0 and 1 for z = 1.96, but we clip
    # them in case of numerical errors or z changing in the future.
    numerator_lower = mu - (z * (var_mean + var_delta)**0.5 + 1)
    numerator_upper = mu + (z * (var_mean - var_delta)**0.5 + 1)
    w_lower = np.maximum(0, numerator_lower / denominator)
    w_upper = np.minimum(1, numerator_upper / denominator)
    w_lower = np.where(
        num_successes == 0, 0,
        np.where(num_successes == num_trials, (alpha/2)**(1/n), w_lower))
    w_upper = np.where(
        num_successes == num_trials, 1,
        np.where(num_successes == 0, 1-(alpha/2)**(1/n), w_upper))
    return w_lower, w_upper

# We did not implement bsc_interval(num_trials, num_successes)
# The future implementation could call
# https://github.com/keithw/biostat/blob/master/blyth-still-casella.cc
# See https://projecteuclid.org/journals/statistical-science/volume-16/issue-2/Interval-Estimation-for-a-Binomial-Proportion/10.1214/ss/1009213286.full (comment by Casella)
