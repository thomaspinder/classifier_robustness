qwertyKeyboardArray = [
    ['`', '1', '2', '3', '4', '5', '6', '7', '8', '9', '0', '-', '='],
    ['q', 'w', 'e', 'r', 't', 'y', 'u', 'i', 'o', 'p', '[', ']', '\\'],
    ['a', 's', 'd', 'f', 'g', 'h', 'j', 'k', 'l', ';', '\''],
    ['z', 'x', 'c', 'v', 'b', 'n', 'm', ',', '.', '/'],
    ['', '', ' ', ' ', ' ', ' ', ' ', '', '']
]

qwertyShiftedKeyboardArray = [
    ['~', '!', '@', '£', '$', '%', '^', '&', '*', '(', ')', '+'],
    ['Q', 'W', 'E', 'R', 'T', 'Y', 'U', 'I', 'O', 'P', '{', '}', '|'],
    ['A', 'S', 'D', 'F', 'G', 'H', 'J', 'K', 'L', ':', '"'],
    ['Z', 'X', 'C', 'V', 'B', 'N', 'M', '<', '>', '?'],
    ['', '', ' ', ' ', ' ', ' ', ' ', '', '']
]

layoutDict = {'QWERTY': (qwertyKeyboardArray, qwertyShiftedKeyboardArray)}

# Sets the default keyboard to use to be QWERTY.
keyboardArray = qwertyKeyboardArray
shiftedKeyboardArray = qwertyShiftedKeyboardArray

def arrayForChar(c):
    if (True in [c in r for r in keyboardArray]):
        return keyboardArray
    elif (True in [c in r for r in shiftedKeyboardArray]):
        return shiftedKeyboardArray
    else:
        return 0

# Finds a 2-tuple representing c's position on the given keyboard array.  If
# the character is not in the given array, throws a ValueError
def getCharacterCoord(c, array):
    if array == 0:
        return 0
    else:
        for r in array:
            if c in r:
                row = array.index(r)
                column = r.index(c)
                return (row, column)
        raise ValueError(c + " not found in given keyboard layout")


# Finds the Euclidean distance between two characters, regardless of whether
# they're shifted or not.
def euclideanKeyboardDistance(c1, c2):
    coord1 = getCharacterCoord(c1, arrayForChar(c1))
    coord2 = getCharacterCoord(c2, arrayForChar(c2))
    if coord1==0 or coord2==0:
        return 10
    else:
        return ((coord1[0] - coord2[0]) ** 2 + (coord1[1] - coord2[1]) ** 2) ** (0.5)