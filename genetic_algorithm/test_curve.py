import pytest
import importlib
import numpy as np

curve = importlib.import_module("02_curve")


def test_binary_convertion():
    response = curve.convert_to_binary(100)
    assert '01100100' == response

def test_binary_convertion_has_8_chars_length():
    response = curve.convert_to_binary(1)
    assert '00000001' == response

def test_split_value_in_3_bits_upper_5_lower():
    BINARY_VALUE = '10110000'
    upper_length = 3
    upper = str()
    lower = str()
    upper, lower = curve.split_binary_value(upper_length, BINARY_VALUE)
    assert '101' == upper
    assert '10000' == lower

def test_select_random_bits_to_cut():
    response = curve.select_random_bits_to_cut()
    assert response >= 1 and response <= 56

def test_select_cut_position():
    response = curve.select_cromosome_cut_position(21)
    assert response == (2, 0.625)
    response = curve.select_cromosome_cut_position(26)
    assert response == (3, 0.25)
    response = curve.select_cromosome_cut_position(36)
    assert response == (4, 0.5)
    response = curve.select_cromosome_cut_position(8)
    assert response == (0, None)
    response = curve.select_cromosome_cut_position(24)
    assert response == (2, None)

def test_cut_is_not_clean():
    bits_to_cut = 17
    response = curve.is_cut_clean(bits_to_cut)
    assert response == False

def test_cut_is_clean():
    bits_to_cut = 24
    response = curve.is_cut_clean(bits_to_cut)
    assert response == True


def test_split_gene():
    cromosome = [165, 95, 130, 250, 140, 27, 133]
    index_to_cut = 2
    bits_to_cut = 3
    # Should pick index 2 which is value 24 in father_cromosome list
    # 24 0.625
    # Binary value 0001 1000 and 0.625 * 8 is 5 digits to cut
    response = curve.split_gene(cromosome, index_to_cut, bits_to_cut)
    assert response[0] == '100'
    assert response[1] == '00010'


def test_reproduction_1():
    total_bits_to_cut = 19
    father_cromosome = [165, 95, 130, 250, 140, 27, 133]
    mother_cromosome = [187, 186, 224, 163, 238, 98, 198]
    expected_child_1 = [165, 95, 128, 163, 238, 98, 198]
    expected_child_2 = [187, 186, 226, 250, 140, 27, 133]
    child_1, child_2 = curve.reproduce(father_cromosome, mother_cromosome, total_bits_to_cut)
    print("child1")
    print(child_1)
    print("expected1")
    print(expected_child_1)
    assert np.array_equal(child_1, expected_child_1) == True
    assert np.array_equal(child_2, expected_child_2) == True

def test_reproduction_2():
    total_bits_to_cut = 24
    father_cromosome = [91, 71, 110, 14, 190, 53, 61]
    mother_cromosome = [187, 186, 224, 163, 238, 98, 198]
    expected_child_1 = [165, 95, 128, 163, 238, 98, 198]
    expected_child_2 = [187, 186, 226, 250, 140, 27, 133]
    child_1, child_2 = curve.reproduce(father_cromosome, mother_cromosome, total_bits_to_cut)
    print("child1")
    print(child_1)
    print("expected1")
    print(expected_child_1)
    assert np.array_equal(child_1, expected_child_1) == True
    assert np.array_equal(child_2, expected_child_2) == True