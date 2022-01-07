# core
import re
import sys


def printEmojis(s: str) -> None:
	'''
	Checks whether or not the operating system is mac or linux.
	If so, emojis are printed as normal, else they are filtered from the string.
	'''

	if sys.platform not in ["linux", "darwin"]:
		print(s)
	else:
		regex = re.compile(
			'['
			u'\U00002600-\U000026FF'  # miscellaneous
			u'\U00002700-\U000027BF'  # dingbats
			u'\U0001F1E0-\U0001F1FF'  # flags (iOS)
			u'\U0001F600-\U0001F64F'  # emoticons
			u'\U0001F300-\U0001F5FF'  # symbols & pictographs I
			u'\U0001F680-\U0001F6FF'  # transport & map symbols
			u'\U0001F900-\U0001F9FF'  # symbols & pictographs II
			u'\U0001FA70-\U0001FAFF'  # symbols & pictographs III
			']+',
			flags=re.UNICODE,
		)
		print(regex.sub(r'', s).strip())
