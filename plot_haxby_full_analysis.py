"""In this script we reproduce the original data analysis conducted by
Haxby et al. in 

"Distributed and Overlapping Representations of Faces and Objects
    in Ventral Temporal Cortex"

(Science 2001)

"""


### Fetch data using nilearn dataset fetcher ################################
# specify how many subjects we want to load
n_subjects = 2

from nilearn import datasets
data_files = datasets.fetch_haxby(n_subjects=n_subjects, fetch_stimuli=True)

### Do analysis for all subjects in subjects list ##########
subject_ids = [1]

# Load nilearn NiftiMasker, the practical masking and unmasking tool
from nilearn.input_data import NiftiMasker
import numpy as np


for subject_id in subject_ids:

    # load labels
    labels = np.recfromcsv(data_files.session_target[subject_id],
                           delimiter=" ")
    # identify resting state labels in order to be able to remove them
    resting_state = labels['labels'] == "rest"

    # find names of remaining active labels
    unique_labels = np.unique(labels['labels'][resting_state == False])

    # extract tags indicating to which acquisition run a tag belongs
    session_labels = labels["chunks"][resting_state == False]


    ### Let us now check the provided masks and do decoding on them

    # Make a data splitting object for cross validation
    from sklearn.cross_validation import LeaveOneLabelOut
    outer_cv = LeaveOneLabelOut(session_labels)

    from sklearn.svm import SVC
    from sklearn.multiclass import OneVsRestClassifier
    from sklearn.cross_validation import cross_val_score
    # classifier = OneVsRestClassifier(SVC(C=.1, kernel="linear"))


    mask_names = ['mask_vt', 'mask_face', 'mask_face_little',
                  'mask_house', 'mask_house_little']

    classifier_penalties = np.logspace(-3, 3, 7)

    mask_scores = {}
    for mask_name in mask_names:
        print "Working on mask %s" % mask_name
        masker = NiftiMasker(data_files[mask_name][subject_id])
        masked_timecourses = masker.fit_transform(
            data_files.func[subject_id])[resting_state == False]

        mask_scores[mask_name] = {}

        for label in unique_labels:
            print "Treating %s %s" % (mask_name, label)
            classification_target = \
                labels['labels'][resting_state == False] == label
            mask_scores[mask_name][label] = {}
            for i, (train, test) in enumerate(outer_cv):
                train_data = masked_timecourses[train]
                train_targets = classification_target[train]
                remaining_labels = session_labels[train]
                inner_cv = LeaveOneLabelOut(remaining_labels)
                mask_scores[mask_name][label][i] = {}
                for C in classifier_penalties:
                    print "Treating %s %s fold %d penalty %1.2f" % (
                        mask_name, label, i, C)
                    mask_scores[mask_name][label][i][C] = cross_val_score(
                        SVC(C=C), 
                        masked_timecourses,
                        classification_target,
                        cv=inner_cv,
                        n_jobs=1,
                        verbose=True,
                        scoring="f1")

                    print "Scores: %1.2f +- %1.2f" % (
                        mask_scores[mask_name][label][i][C].mean(),
                        mask_scores[mask_name][label][i][C].std())

    # make a rudimentary diagram
    import matplotlib.pyplot as plt
    score_means = np.array([[mask_scores[mask_name][label].mean()
                for label in unique_labels] 
                for mask_name in mask_names])
    plt.matshow(score_means)
    plt.xticks(range(len(unique_labels)), unique_labels, rotation=90)
    plt.yticks(range(len(mask_names)), mask_names)
    plt.colorbar()
    plt.hot()
    plt.show()


