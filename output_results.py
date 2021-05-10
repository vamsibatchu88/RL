from typing import List, Any, Union


def round_all_values(values: List[Any], num_decimals: int=2):
    """
    Rounds all the values in the given structure to the given number of decimal points.

    :param values: The list of things to round
    :param num_decimals: The number of decimal values to round to
    """
    for i, v in enumerate(values):
        if type(v) is list:
            round_all_values(v)
        else:
            values[i] = round(v, num_decimals)


def single_experiment_as_string(results: List[List[Union[str, float]]]):
    """
    Gets the results as a comma-separated string.

    :param results: The list of experiment results
    :return: A string representation of the given results
    """
    output_str = "run_num"

    num_steps = len(results[0])
    for i in range(1, num_steps + 1):
        output_str += ",step_" + str(i)
    output_str += "\n"

    for i, results in enumerate(results):
        output_str += str(i + 1)
        for value in results:
            output_str += "," + str(value)
        output_str += "\n"
    return output_str


def multiple_experiments_as_string(all_results: List[List[List[Union[str, float]]]], test_names: List[str]):
    """
    Gets the results as a comma-separated string.

    :param all_results: The list of experiment results
    :param test_names: The name for each test
    :return: A string representation of the given results
    """
    assert len(test_names) == len(all_results)

    # Adds the header row
    output_str = "test_name,run_num"

    num_steps = len(all_results[0][0])
    for i in range(1, num_steps + 1):
        output_str += ",step_" + str(i)
    output_str += "\n"

    for i, test_results in enumerate(all_results):
        for j, run_results in enumerate(test_results):
            assert len(run_results) == num_steps
            output_str += str(test_names[i]) + ","  # adds test name
            output_str += str(j + 1)  # adds run number
            for value in run_results:
                output_str += "," + str(value)
            output_str += "\n"
    return output_str
