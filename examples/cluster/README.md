# Cluster submission

This example shows how to run Q-value calculation on clusters using [HTCondor](https://htcondor.org/).

## Getting started

Set up a virtual environment in this directory:

```
#create virtual environment
python3 -m venv env
#activate virtual environment
source env/bin/activate
#install mouselab package, which includes all imported packages
pip install ../..
```

## Submitting a job

First, create the log folder in this directory:

```
mkdir log
```

From this directory, with bidding:

````
condor_submit_bid <bid_amount> submission_scripts/q_value_calculation.sub  experiment_setting=<experiment_setting>
````

For example:

````
condor_submit_bid 2 submission_scripts/q_value_calculation.sub  experiment_setting=high_increasing
````