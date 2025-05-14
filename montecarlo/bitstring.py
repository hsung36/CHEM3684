import numpy as np

class BitString:
    """
    Simple class to implement a config of bits
    """
    class bit:
        def __init__(self, num):
            self.num = 0

        def __str__(self):
            return "%10s"%(self.num)


    def __init__(self, N):
        self.N = N
        self.config = np.zeros(N, dtype=int) 

    def __repr__(self):
        out = ""
        for i in self.config:
            out += str(i)
        return out

    def __eq__(self, other):        
        return all(self.config == other.config)
    
    def __len__(self):
        return len(self.config)

    def on(self):
        """
        Return number of bits that are on
        """
        on = 0
        for i in self.config:
            if i == 1:
                on += 1
        return on


    def off(self):
        """
        Return number of bits that are on
        """
        off = 0
        for i in self.config:
            if i == 0:
                off += 1
        return off

    def flip_site(self,i):
        """
        Flip the bit at site i
        """
        if self.config[i] == 0:
            self.config[i] = 1
        else:
            self.config[i] = 0

    
    def integer(self):
        """
        Return the decimal integer corresponding to BitString
        """
        i = 0
        sum = 0
        length = len(self.config)

        while length > i:
            sum += self.config[length - i - 1] * (2 ** (i))
            i += 1
        
        return sum


    def set_config(self, s:list[int]):
        """
        Set the config from a list of integers
        """
        self.config = s

    def set_integer_config(self, dec:int):
        """
        convert a decimal integer to binary
    
        Parameters
        ----------
        dec    : int
            input integer
            
        Returns
        -------
        Bitconfig
        """
        num = dec
        i = len(self.config)
        self.config = np.zeros(i, dtype=int)
        while num > 0:
            self.config[i - 1] = num % 2
            num = num // 2
            i -= 1