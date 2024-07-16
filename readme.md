# Trace Server Protocol (TSP) Machine Learning Library (TMLL)

## Installation
Use the following command to install TMLL:
```console
pip3 install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple tmll
```

## Usage
In order to use TMLL, you need to import its client in your code.
```python
from tmll import TMLLClient

client = TMLLClient()
```

The client connects to the running TSP server with the default hostname and ports (i.e., localhost:8080). However, if the TSP server is not running or its health status is not stable, the client will raise a connection error.

### Importing Traces
To import your traces in TMLL, you can use `import_traces()` method.
```python
client = TMLLClient()

client.import_traces(traces = [
    {
        "path": "YOUR_TRACE_FILE_PATH_1",
        "name": "YOUR_TRACE_NAME_1" # This is optional. If you pass the generate_name=True to the method, the client will automatically create a name for your traces (and for the experiment if neccessary)
    },
    {
        "path": "YOUR_TRACE_FILE_PATH_2",
        "name": "YOUR_TRACE_NAME_2" # Check above
    },
    ...
])
```

### Apply Clustering
```python
clustering = client.apply_clustering(with_results=True)

if clustering:
    # Experiment info
    experiment = clustering["experiment"]
    print(f"Experiment: {experiment}")

    # Fetched outputs from TSP
    outputs = clustering["outputs"]
    for output in outputs:
        print(f"Output: {output['output'].name}")

        # If clustering has been applied successfully on the output's data
        if "results" in output:
            clusters = output["results"]

            for cluster_name, cluster_info in clusters.items():
                print(f"Cluster: {cluster_name}")
                print(f"\tModel: {cluster_info['model']} | Number of Clusters: {cluster_info['n_clusters']} | Evaluations: {cluster_info['evaluation']}")

                dataframe = cluster_info["clusters"]
```