#Danny Hong
#ECE 467 Project 2
#Spring 2021

import sys

#This function creates an output string using bracket notation for the parse
def create_parse_text(node):

    if node[2] is None:
        output_string = "[" + node[0] + " " + node[1] + "]"
        return output_string

    else:
        output_string = "[" + node[0] + " " + create_parse_text(node[1]) + " " + create_parse_text(node[2]) + "]"
        return output_string

#This function prints the parse as a tree
def print_parse_tree(node, level):

    for i in range(0, level):
        sys.stdout.write("  ")

    if node[2] is None:
        sys.stdout.write(node[0] + " ")
        sys.stdout.write(node[1] + "\n")

    else:
        sys.stdout.write(node[0] + "\n")
        print_parse_tree(node[1], level + 1)
        print_parse_tree(node[2], level + 1)

def main():

    print("\nLoading grammar...")

    grammar_rules = open(sys.argv[1], "r").read().splitlines()
    display_parse_trees = input("Do you want textual parse trees to be displayed (y/n)?: ")

    # Combines the grammar rules into a list
	# For instance:
	#	A --> B C
	#	C --> D
	# turns into: [ [A, B, C], [C, D] ]
    grammar_rules_list = []
    for grammar_rule in grammar_rules:
        grammar_rule = grammar_rule.split()
        grammar_rule.remove('-->')
        grammar_rules_list.append(grammar_rule)

	#The parser program will continue to run until the user types "quit"
    while True:

        sentence = input("Enter a sentence: ")

        if sentence == "quit":
            print("Goodbye!")
            exit(0)

        #Spliting the inputted sentence into tokens
        tokens = sentence.split()

    	#Generating an empty table for parsing
        table = [[[] for i in range(len(tokens) - j)] for j in range(len(tokens))]

		#Populating the table with nodes containing terminals in the first row
        index = 0

        for token in tokens:
            for grammar_rule in grammar_rules_list:
                if token == grammar_rule[1]:
                    table[0][index].append([grammar_rule[0], token, None])
            index = index + 1

		#Implementing the CKY Algorithm (Source: en.wikipedia.org/wiki/CYK_algorithm)
        for remaining_words in range(2, len(tokens) + 1): 
            for cell in range(0, (len(tokens) - remaining_words) + 1):
                for left_partition in range(1, remaining_words):
                    right_partition = remaining_words - left_partition
                    left_cell = table[left_partition - 1][cell]
                    right_cell = table[right_partition - 1][cell + left_partition]

                    for grammar_rule in grammar_rules_list:

                        l_nodes_list, r_nodes_list = [], []

                        for node in left_cell:
                            if node[0] == grammar_rule[1]:
                                l_nodes_list.append(node)

                        if l_nodes_list:
                            for node in right_cell:
                                if len(grammar_rule) == 3 and node[0] == grammar_rule[2]:
                                    r_nodes_list.append(node)

                            for ln in l_nodes_list:
                                for rn in r_nodes_list:
                                    table[remaining_words - 1][cell].append([grammar_rule[0], ln, rn])

        
		#Reading the parse table in reverse from bottom to top
        nodes_list = []
        for node in table[-1][0]:	
            if node[0] == "S":
                nodes_list.append(node)

        #The parse is valid if the nodes_list is not an empty set 
        if nodes_list:
            print("VALID SENTENCE")

            parse_count = 0
            iterator = 1

            for node in nodes_list:               
                print("\nValid Parse #" + str(iterator) + ": ")
                print(create_parse_text(node) + "\n")

                if display_parse_trees == "y":
                    print_parse_tree(node, 0)
                iterator = iterator + 1
                parse_count = parse_count + 1

            print("\nNumber of Valid Parses:", parse_count) 

        else:
            print("\nNO VALID PARSES\n")

if __name__ == "__main__":
    main()
    