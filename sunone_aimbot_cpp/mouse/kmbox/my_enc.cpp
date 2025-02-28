void my_encrypt(unsigned char* input, unsigned char* key)
{	unsigned char  n =32;
	unsigned long* a1 = (unsigned long*)input;
	unsigned long* a2 = (unsigned long*)key;
	unsigned long  a3 = a1[n - 1], a4 = a1[0], sum = 0, a5;
	unsigned char a6, a7;
	a7 = 6;//łőĘźÖľ
	while (a7-- > 0)
	{
		sum += 2654435769; 
		a5 = sum >> 2 & 3;
		for (a6 = 0;a6 < n - 1; a6++)
			a4 = a1[a6 + 1],
			a3 = a1[a6] += (a3 >> 5 ^ a4 << 2) + (a4 >> 3 ^ a3 << 4) ^ (sum ^ a4) + (a2[a6 & 3 ^ a5] ^ a3);
		a4 = a1[0];
		a3 = a1[n - 1] += (a3 >> 5 ^ a4 << 2) + (a4 >> 3 ^ a3 << 4) ^ (sum ^ a4) + (a2[a6 & 3 ^ a5] ^ a3);
	}
}
