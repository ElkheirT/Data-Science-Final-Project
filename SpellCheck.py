import pkg_resources
from symspellpy.symspellpy import SymSpell, Verbosity

dictionary_path = pkg_resources.resource_filename("symspellpy", "frequency_dictionary_en_82_765.txt")
spellchecker = SymSpell()
spellchecker.load_dictionary(dictionary_path, term_index=0, count_index=1)

def get_correction(word):
    if word not in spellchecker.words.keys():
        suggestions = spellchecker.lookup(word, Verbosity.TOP, 2)  # Returns the top suggestions first
        if len(suggestions) > 0:
            return suggestions[0].term
    return word



