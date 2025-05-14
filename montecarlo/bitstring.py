import numpy as np

class BitString:
    """
    Simple class to implement a configuration of bits.

    This class represents a sequence of bits (0s and 1s) and provides
    methods to manipulate and query the bit configuration.

    Attributes
    ----------
    N : int
        The number of bits in the string.
    config : numpy.ndarray
        A 1D NumPy array of integers (0 or 1) representing the bit configuration.
    """
    class bit: # This inner class seems unused in the current BitString implementation.
              # If it's intended for use, it should also have a docstring.
        def __init__(self, num):
            """
            Initializes a bit object.

            Note: This inner class 'bit' is not currently used by the BitString class.
                  Its purpose and interaction with BitString are unclear.

            Parameters
            ----------
            num : int
                The initial numerical value for the bit (currently unused effectively).
            """
            self.num = 0 # This seems to always be 0, consider its purpose.

        def __str__(self):
            """
            Returns a string representation of the bit object.

            Note: This inner class 'bit' is not currently used by the BitString class.

            Returns
            -------
            str
                A string representation of the 'num' attribute, formatted to 10 characters.
            """
            return "%10s"%(self.num)


    def __init__(self, N):
        """
        Initializes a BitString object of length N with all bits set to 0.

        Parameters
        ----------
        N : int
            The number of bits in the string.
        """
        self.N = N
        self.config = np.zeros(N, dtype=int)

    def __repr__(self):
        """
        Returns an 'official' string representation of the BitString object.

        This representation is a string of 0s and 1s.

        Returns
        -------
        str
            The string representation of the bit configuration.
        """
        out = ""
        for i in self.config:
            out += str(i)
        return out

    def __eq__(self, other):
        """
        Checks if this BitString is equal to another BitString.

        Two BitStrings are considered equal if their configurations are identical.

        Parameters
        ----------
        other : BitString
            The other BitString object to compare against.

        Returns
        -------
        bool
            True if the configurations are equal, False otherwise.
        """
        if not isinstance(other, BitString):
            return NotImplemented
        return all(self.config == other.config)

    def __len__(self):
        """
        Returns the length of the BitString (number of bits).

        Returns
        -------
        int
            The number of bits in the configuration.
        """
        return len(self.config)

    def on(self):
        """
        Return number of bits that are on (i.e., set to 1).

        Returns
        -------
        int
            The count of bits that are 1.
        """
        on_count = 0 # Renamed variable to avoid conflict with method name
        for bit_val in self.config: # Renamed loop variable for clarity
            if bit_val == 1:
                on_count += 1
        return on_count


    def off(self):
        """
        Return number of bits that are off (i.e., set to 0).

        Returns
        -------
        int
            The count of bits that are 0.
        """
        off_count = 0 # Renamed variable
        for bit_val in self.config: # Renamed loop variable
            if bit_val == 0:
                off_count += 1
        return off_count

    def flip_site(self, i):
        """
        Flip the bit at site i (0 becomes 1, 1 becomes 0).

        Parameters
        ----------
        i : int
            The index of the bit to flip.
        """
        if self.config[i] == 0:
            self.config[i] = 1
        else:
            self.config[i] = 0


    def integer(self):
        """
        Return the decimal integer corresponding to the BitString.

        The BitString is interpreted as a binary number, with the bit at index 0
        being the least significant bit (LSB) if read from right to left,
        or most significant if read from left to right as per typical array indexing.
        This implementation assumes standard binary-to-decimal conversion
        where config[N-1] is 2^0, config[N-2] is 2^1, ..., config[0] is 2^(N-1).

        Returns
        -------
        int
            The decimal representation of the bit string.
        """
        current_sum = 0 # Renamed variable to avoid conflict with sum() built-in
        power = 0
        # Iterate from the last bit (least significant in this interpretation)
        for i in range(len(self.config) - 1, -1, -1):
            current_sum += self.config[i] * (2 ** power)
            power += 1
        return current_sum


    def set_config(self, s: list[int]):
        """
        Set the config from a list of integers (0s or 1s).

        The length of the input list must match the BitString's length (N).

        Parameters
        ----------
        s : list[int]
            A list of integers (0 or 1) to set as the new configuration.
            The length of this list should be equal to self.N.

        Raises
        ------
        ValueError
            If the length of the input list `s` does not match `self.N`.
        """
        if len(s) != self.N:
            raise ValueError(f"Input list length {len(s)} must match BitString length {self.N}")
        self.config = np.array(s, dtype=int) # Ensure it's a numpy array of int

    def set_integer_config(self, dec: int):
        """
        Set the BitString's configuration from a decimal integer.

        Converts the decimal integer to its binary representation and sets
        the BitString's bits accordingly. The binary representation will be
        padded with leading zeros if necessary to fit the BitString's length N.
        If the integer is too large to be represented by N bits,
        only the N least significant bits will be used.

        Parameters
        ----------
        dec : int
            The input decimal integer. Must be non-negative.

        Raises
        ------
        ValueError
            If the input integer `dec` is negative.
        """
        if dec < 0:
            raise ValueError("Input integer must be non-negative.")

        num = dec
        new_config = np.zeros(self.N, dtype=int)
        i = self.N - 1 # Start from the last index (least significant bit)
        while num > 0 and i >= 0:
            new_config[i] = num % 2
            num = num // 2
            i -= 1
        self.config = new_config