import logging
import doctest
import argparse


def merge(input_list, left_part, right_part):
    i, j, k = 0, 0, 0

    while i < len(left_part) and j < len(right_part):
        if left_part[i] <= right_part[j]:
            logging.info('Adding val {} to input_list checking new one'.format(left_part[i]))
            input_list[k] = left_part[i]
            i = i + 1
        else:
            logging.info('Adding val {} to input_list checking new one'.format(right_part[j]))
            input_list[k] = right_part[j]
            j = j + 1
        k = k + 1

    while i < len(left_part):
        logging.info('Adding val {} to input_list checking new one'.format(left_part[i]))
        input_list[k] = left_part[i]
        i = i + 1
        k = k + 1

    while j < len(right_part):
        logging.info('Adding val {} to input_list checking new one'.format(right_part[j]))
        input_list[k] = right_part[j]
        j = j + 1
        k = k + 1
    logging.info('Returning list {} and left, right part L:{}, R:{}'.format(input_list, left_part, right_part))
    return input_list, left_part, right_part


def merge_sort(input_list):
    """Merge sort is an efficient, general-purpose, comparison-based sorting algorithm.
    Most implementations produce a stable sort,
    which means that the order of equal elements is the same in the input and output.

    Takes one parameter returns sorted list

        >>> merge_sort([15, 13, 11, 16, 18, 69, 46, 22, 11, 10, 8, 5, 8])
        [5, 8, 8, 10, 11, 11, 13, 15, 16, 18, 22, 46, 69]
        >>> merge_sort([15, 13, 11, 16, 18, 69, 46, 22, 11, 10, 8, 5, 9])
        [5, 8, 9, 10, 11, 11, 13, 15, 16, 18, 22, 46, 69]
        """
    if input_list:
        logging.info('Input list exists {}'.format(input_list))
        first_el_type = type(input_list[0])
        if not all(isinstance(x, first_el_type) for x in input_list):
            logging.warning("List of multiple types can't be sorted {}".format(input_list))
            raise ValueError("List of multiple types can't be sorted")
        else:
            if len(input_list) > 1:
                logging.info('Does list length greater 1 - {} -{}'.format(len(input_list) > 1, len(input_list)))
                left_part = input_list[:len(input_list) // 2]
                right_part = input_list[len(input_list) // 2:]
                logging.info('Left right part before merge sort- L:{}, R:{}'.format(left_part, right_part))

                merge_sort(left_part)
                merge_sort(right_part)

                logging.info('Left right part after merge sort- L:{}, R:{}'.format(left_part, right_part))
                merge(input_list, left_part, right_part)
            return input_list

    else:
        return input_list
INPUT_VALUE = [5, 1, 3, 2, 1, 5, 7, 8, 10, 2]
print(merge_sort(INPUT_VALUE))