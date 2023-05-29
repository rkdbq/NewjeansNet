import numpy as np

def map_numbers_to_text(array, mapping_dict):
    # Create a new array for mapped text values
    mapped_array = np.vectorize(mapping_dict.get)(array)

    return mapped_array

def numpy_array_to_csv(array, filename):
    # Create an array with 'id' as index
    array_with_id = np.column_stack((np.arange(len(array)), array))

    # Save the array as CSV
    np.savetxt(filename, array_with_id, delimiter=",", fmt="%s", header="id,label", comments="")