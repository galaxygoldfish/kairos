Big Ideas: Data Loss, Bandpass Filtering, Data Segmentation

Several Changes:
    - diminishing DATA LOSS
        - sample dependent data acquisition
        - increasing max_chunklen in StreamInlet (larger chunks, reducing chance of buffer, slight
            increase in latency)
        - changed SHIFT_LENGTH from 0.1 to 0.25 (increasing makes it less likely to miss samples
            due to processing delays)
    - filtering eeg data to 8-30 (gamma is too hard to detect with Muse because of physical artifacts)
    - live version of data vs. filtered
        - filter the data at the end for best results
    - new csv's

We hadn’t really thought about data loss before, but it’s important. If we lose samples, we might see fake or incorrect frequencies. To fix this, we changed the way data is collected — instead of recording for a set amount of time, we now record until a set number of samples is reached. We also adjusted other settings like max_chunklen to help reduce missed data.

We started filtering the data to focus on the brain wave range we care about (8–30 Hz), which helps detect “locked-in” brain states. This changed how the graphs look because filtering removes extra noise. We realized that filtering live data can cause issues, so the live graph shows raw data (with noise), while the saved graph is filtered and clean. Having both is useful: one lets us monitor the session, and the other gives better data for analysis.

We also now save two CSV files:

Raw time CSV – shows each channel’s signal over time (with timestamps).

Segmented CSV – formatted for machine learning. Each row contains 2 seconds of data from all channels.

This segmentation helps a lot. Instead of giving the ML model one sample from a long session, it gets multiple short, rich samples. This makes it easier to detect small patterns or changes.
