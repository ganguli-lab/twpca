import numpy as np


def jittered_neuron(t=None, feature=None, n_trial=61, jitter=1.0, gain=0.0, noise=0.05, seed=1234):
    """Generates a synthetic dataset of a single neuron with a jittered firing pattern.

    Parameters
    ----------
    t : array_like
        vector of within-trial timepoints
    feature : function
        produces a jittered instance of the feature (takes time shift as an input)
    n_trial : int
        number of trials
    jitter : float
        standard deviation of trial-to-trial shifts
    gain : float
        standard deviation of trial-to-trial changes in amplitude
    noise : float
        scale of additive gaussian noise
    seed : int
        seed for the random number generator

    Returns
    -------
    canonical_feature : array_like
        vector of firing rates on a trial with zero jitter
    aligned_data : array_like
        n_trial x n_time x 1 array of de-jittered noisy data
    jittered_data : array_like
        n_trial x n_time x 1 array of firing rates with jitter and noise
    """

    # default time base
    if t is None:
        t = np.linspace(-5, 5, 150)

    # default feature
    if feature is None:
        feature = lambda tau: np.exp(-(t-tau)**2)

    # noise matrix
    np.random.seed(seed)
    noise = noise*np.random.randn(n_trial, len(t))

    # generate jittered data
    gains = 1.0 + gain*np.random.randn(n_trial)
    shifts = jitter*np.random.randn(n_trial)
    jittered_data = np.array([g*feature(s) for g, s in zip(gains, shifts)]) + noise

    # generate aligned data
    aligned_data = np.array([g*feature(0) for g in gains]) + noise

    return feature(0), np.atleast_3d(aligned_data), np.atleast_3d(jittered_data)


def jittered_population(n_trial=100, n_time=130, n_neuron=50, n_events=3, tau=10., event_gap=25, max_jitter=15):
    """Generates a synthetic spiking dataset of a population of neurons with correlated jitters.

    Parameters
    ----------
    n_trial : int
        number of trials
    n_time : int
        number of within-trial timepoints
    n_neuron : int
        number of recorded neurons
    n_events : int
        number of transient increases in firing rate
    tau : float
        time constant for exponential decay of events
    event_gap : int
        average gap between neural events
    max_jitter : int
        maximum amount of jitter in each event

    Returns
    -------
    rates : array_like
        n_trial x n_time x n_neuron array of firing probabilities
    spikes : array_like
        n_trial x n_time x n_neuron array of observed spikes
    """

    # Randomly generate jitters
    jitters = np.random.randint(-max_jitter, max_jitter, size=(n_trial, n_events))
    ordering = np.argsort(jitters[:, 0])
    jitters = jitters[ordering]

    # Create one-hot matrix that encodes the location of latent events
    events = np.zeros((n_trial, n_time))
    for trial_idx, jitter in enumerate(jitters):
        trial_event_times = np.cumsum(event_gap + jitter)
        events[trial_idx, trial_event_times] = 1.0
    avg_event = np.zeros(n_time)
    avg_event[np.cumsum([event_gap] * n_events)] = 1.0

    # Convolve latent events with an exponential filter
    impulse_response = np.exp(-np.arange(n_time)/float(tau))
    impulse_response /= impulse_response.sum()

    latents = np.array([np.convolve(e, impulse_response, mode='full')[:n_time] for e in events])

    # Coupling from one dimensional latent state to each neuron
    readout_weights = np.random.rand(n_neuron) + 0.1

    # Probability of firing for each neuron
    rates = np.exp(np.array([np.outer(latent, readout_weights) for latent in latents]))
    rates -= rates.min()
    rates /= rates.max()

    # Sample spike trains
    spikes = np.random.binomial(1, rates).astype(np.float32)

    return rates, spikes
