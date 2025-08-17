## Setting up the project to run NN on some data

Things to know:
- `qrennd` expects a file structure that looks like this:

EXP_DIR
    dataset-name-1
        directory-1
            data
        directory-2
            data
        ...
    dataset-name-2
    ...

The dataset-names are usually 'train', 'val', 'test'.
It only goes through the directories inside dataset-name in two loops: 
(1) num_rounds and (2) states.
Therefore, if one also wants to loop over some other thing, one has to rewrite
the function `qrennd.datasets.generators.dataset_generator`.

- `qrennd` does not require anything about the structure of the data.

- One must write a function to preprocess the data for the NN. One can find examples
in `qrennd.datasets.preprocessing`. The structucture of the data must match the
function used. Advice: store 'raw' data and do all the preprocessing on the fly
(e.g. compute defects from outcomes...) because it is then easier to change the
inputs of the NN (e.g. soft-info, concatenate vectors...) and it does not use
so much memory space. 

- Adapt `qrennd.datasets.preprocessing.to_model_inpus` if one needs to concatenate
vectors before giving them to the NN (see the code in the `experimental_data` branch).

- One must adapt the function `qrennd.datasets.util.load_datasets` to use the
correct processing function to the data. 
