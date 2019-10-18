import json

import pandas as pd


class Ingredients():
    """The Ingrediens class contains a list of ingredients and provides access to them. It provides
    functionality to search the list of substances using a search term.

    If it is generated manually, a list of ingredients can be transferred to it. A JSON file is then
    not necessary.

    If it is created using the instance method, it is based on the JSON file defined in json_path.

    The static convert method can be used to convert an Excel file corresponding to the requirements
    into a JSON file. Requirements are: Sheet 1, start at A1 (header): ID, E-Nr, ingrediens (comma separated),
    remark, annotation, classification, keywords (comma separated)

    An ingredient represents a list with the properties specified here:
    ingredient_new = [id, e_number, ingrediens_list, remark, annotation, classification, keywords_list]"""

    def __init__(self, items=[], usePatch=True):
        """The constructor. A list of ingredients can be passed as an optional argument.
        ingredient_new = [id, e_number, ingrediens_list, remark, annotation, classification, keywords_list].

        :param items:An optional list of items to add.
        :param usePatch:If true, special characters contained in search words are converted to normal
        letters (e.g. ö -> o)
        """
        try:
            self.usePatch = usePatch
            self.items = items
            self.search_items = {}
        except:
            print('Error in method {0} in module {1}'.format('init', 'ingrediens.py'))

    @staticmethod
    def instance(json_path, usePatch=True):
        """Returns a new instance of the Ingrediens class based on the JSON file named in json_path.

        :param json_path:The path to the Json file that serves as the database.
        :param usePatch:If true, special characters contained in search words are converted to normal
        letters (e.g. ö -> o).
        :return:An instance of the class Ingridiens.
        """
        try:
            with open(json_path, mode='r', encoding='utf-8') as json_file:
                json_data = json.load(json_file)
                ingrediens = Ingredients(json_data['items'], usePatch)
                ingrediens.update()

            return ingrediens
        except:
            print('Error in method {0} in module {1}'.format('instance', 'ingrediens.py'))
            return None

    @staticmethod
    def convert(excel_path, json_path):
        """A static method which converts an Excel file specified in excel_path to the JSON file defined by json_path.

        :param excel_path:The excel path.
        :param json_path:The json path.
        """
        try:
            raw = pd.read_excel(excel_path)
            ingrediens_data = Ingredients()

            for x in raw.values:
                id = x[0]
                e_number = str(x[1]).strip()

                ingrediens = str(x[2]).split(',')
                ingrediens_list = [x.strip() for x in ingrediens]

                remark = str(x[3])
                annotation = str(x[4])
                classification = str(x[5])

                keywords = str(x[6]).split(',')
                keywords_list = [x.strip() for x in keywords]

                ingredient_new = [id, e_number, ingrediens_list, remark, annotation, classification, keywords_list]
                ingrediens_data.add(ingredient_new)

            ingrediens_data.update()

            with open(json_path, mode='w', encoding='utf-8') as json_file:
                json.dump(ingrediens_data.__dict__, json_file, ensure_ascii=False)
        except:
            print('Error in method {0} in module {1}'.format('convert', 'ingrediens.py'))

    def __iter__(self):
        """Returns an iterator for the list of ingredients.

        :return:An iterator.
        """
        try:
            return self
        except:
            print('Error in method {0} in module {1}'.format('iter', 'ingrediens.py'))
            return None

    def next(self):
        """Goes through the list of ingredients (Generator).

        :return:The next Item
        """
        try:
            for item in self.items:
                yield item
        except:
            print('Error in method {0} in module {1}'.format('next', 'ingrediens.py'))
            return None

    def add(self, item):
        """Adds a new ingredient to the list of ingredients.
        ingredient_new = [id, e_number, ingrediens_list, remark, annotation, classification, keywords_list].

        :param item:The element to be added.
        """
        try:
            self.items.append(item)
        except:
            print('Error in method {0} in module {1}'.format('add', 'ingrediens.py'))

    def contains(self, item):
        """Returns True and the ID of an ingredient if the transfer item could be assigned to a substance.
        Otherwise, False and -1 are returned.

        During preprocessing, the search string is converted to lowercase letters and all spaces are removed.

        :param item:The element (searchstring) for which a check is to be made.
        :return:True and the ID if exists, otherwise False.
        """
        try:
            item = item.lower()
            item = str(item).replace(' ', '')

            if self.usePatch == True:
                item = self.replaceChar(item)

            if item in self.search_items:
                id = self.search_items[item]
                return True, id
            else:
                return False, -1
        except:
            print('Error in method {0} in module {1}'.format('contains', 'ingrediens.py'))
            return None

    def replaceChar(self, item):
        """Replaces the special characters ü, ö, ä with u, o, a

        :param item:The element (searchstring) for which a check is to be made.
        :return:The item with the replacements
        """
        try:
            str(item).replace("ö", "o")
            str(item).replace("ä", "a")
            str(item).replace("ü", "u")

            return item
        except:
            print('Error in method {0} in module {1}'.format('replaceChar', 'ingrediens.py'))
            return None

    def replaceChar_in_String(self, item):
        """Searches the passed word for the letters ä, ö, ü.

        :param item:The word to analyze
        :return:True, if one of the umlauts was found
        """
        try:
            search_list = ["ä", "ö", "ü"]

            for x in search_list:
                if item.find(x):
                    return True

            return False
        except:
            print('Error in method {0} in module {1}'.format('replaceChar_in_String', 'ingrediens.py'))
            return None

    def get_item(self, id):
        """Returns an ingredient based on the ID of the substance.
        ingredient_new = [id, e_number, ingrediens_list, remark, annotation, classification, keywords_list].

        :param id:The ID for which an element is to be returned.
        :return:The associated element.
        """
        try:
            x = [x for x in self.items if x[0] == id]
            return x
        except:
            print('Error in method {0} in module {1}'.format('get_item', 'ingrediens.py'))
            return None

    def get_enumber(self, id):
        """Returns the name of an ingredient based on the ID of the substance.

        :param id:The Id for which an E-number is to be returned.
        :return:The corresponding E-number.
        """
        try:
            return self.get_item(id)[0][1]
        except:
            print('Error in method {0} in module {1}'.format('get_enumber', 'ingrediens.py'))
            return None

    def get_name(self, id):
        """Returns the name of an ingredient based on the ID of the substance.

        :param id:The ID for which a name is to be returned.
        :return:The corresponding name.
        """
        try:
            return self.get_item(id)[0][2]
        except:
            print('Error in method {0} in module {1}'.format('get_name', 'ingrediens.py'))
            return None

    def get_remark(self, id):
        """Returns the remark assigned to an ingredient based on the ID of the substance.

        :param id:The ID for which a comment is to be returned.
        :return:The corresponding remark.
        """
        try:
            return self.get_item(id)[0][3]
        except:
            print('Error in method {0} in module {1}'.format('get_remark', 'ingrediens.py'))
            return None

    def update(self):
        """Builds a dictionary based on the existing data. The search terms represent the key values, while
        the value represents the ID of the ingredient.

        The list consists of a variation of the E-number (E 123 ==> e123, e-123), the substance
        names and the extended keyword list. If the ingredient or keyword contains an umlaut, a second version
        with the underlying letter is added (a instead of ä etc.).

        Preprocessing converts the matching strings to lowercase letters and removes all blanks during the
        search.
        """
        try:
            x = [x for x in self.items]

            for x in self.items:
                # E-numbers can have attached letters. These must be retained.
                e_number = str(x[1]).lower().strip()
                e_number = e_number[0].replace('e', '') + e_number[1:]
                e_number = e_number.strip()

                self.search_items.update({'e' + e_number: x[0]})
                self.search_items.update({'e-' + e_number: x[0]})

                for y in x[2]:
                    insert_string = str(y).strip().lower().replace(' ', '')
                    self.search_items.update({insert_string: x[0]})

                    if self.replaceChar_in_String(insert_string) == True:
                        self.search_items.update({self.replaceChar(insert_string): x[0]})

                for y in x[6]:
                    insert_srting = str(y).strip().lower().replace(' ', '')
                    self.search_items.update({insert_srting: x[0]})

                    if self.replaceChar_in_String(insert_string) == True:
                        self.search_items.update({self.replaceChar(insert_string): x[0]})
        except:
            print('Error in method {0} in module {1}'.format('update', 'ingrediens.py'))
