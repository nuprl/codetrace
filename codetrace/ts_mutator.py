import tree_sitter
from codetrace.parsing_utils import (
    get_captures,TS_PARSER, typescript_builtin_objects
)
from codetrace.base_mutator import AbstractMutator, MutationFn
import random
from typing import List, Tuple, Optional
import random
"""
Random mutation code.

Some considerations.

1. renaming to an arbitrary name (especially length)

the trick to being able to rename to any name is to accumulate
all the changes and apply them finally from end to start

2. each mutation method should produce different types of names to prevent overlap
and also for semantics
"""
TS_IDENTIFIER_QUERY = """((identifier) @name)"""
TS_TYPE_IDENTIFIER_QUERY = """((type_identifier) @name)"""
TS_PROPERTY_IDENTIFIER_QUERY = """((property_identifier) @name)"""
TS_PREDEFINED_TYPE_QUERY = """((predefined_type) @name)"""
 
TS_VARIABLE_DECLARATION_QUERY = """
(required_parameter pattern: (identifier) @name)
(variable_declarator (identifier) @name)
(function_declaration (identifier) @name)
"""

TS_TYPE_ANNOTATIONS_QUERY = """((type_annotation) @name)"""

TS_PARAM_TYPES_QUERY = """
(required_parameter pattern: (_) (type_annotation) @name)
(optional_parameter pattern: (_) (type_annotation) @name)
return_type: (type_annotation) @name
"""

class TsMutator(AbstractMutator):

    def format_type_alias(self, type_capture: tree_sitter.Node, alias: bytes) -> bytes:
        if type_capture.text.decode("utf-8") in typescript_builtin_objects:
            prefix = b"class " + alias + b" extends " + type_capture.text + b" {};"
        elif type_capture.type == "predefined_type":
            prefix = b"type " + alias + b" = " + type_capture.text + b";"
        else:
            prefix = None
        return prefix

    def add_aliases_to_program(self, program: bytes, aliases: List[bytes]) -> bytes:
        if len(aliases) > 0:
            aliases = b"\n".join(set(aliases)) + b"\n\n"
            program = aliases + program
        return program

    def extract_type_from_annotation(self,node_capture: tree_sitter.Node) -> tree_sitter.Node:
        return node_capture.child(1)
    
    def is_constructor_param(self, x: tree_sitter.Node):
        """
        method_definition
            name: property_identifier
            parameters: formal_parameters
                required_parameter or optional_parameter
                    identifier
        """
        try:
            parent_type = x.parent.parent.parent.type
        except:
            return False
        
        if parent_type != "method_definition":
            return False
        
        name = x.parent.parent.parent.children_by_field_name("name")
        if len(name) > 0:
            assert len(name) == 1
            return name[0].text == b"constructor"
        return False

    def random_mutate(
        self,
        program: str,
        fim_type: str,
        mutations: List[MutationFn],
        debug_seed: Optional[int] = None
    ) -> str:
        """
        Apply random combination of mutations to the program.
        Can provide a random seed DEBUG_SEED for debugging.
        NOTE: if debug_seed is -1, this is a special case where we do not select a random subset but
        and run the full set instead (DEGUB only)
        """
        random.seed(debug_seed)
        # to prevent tree-sitter error:
        program = self.replace_placeholder(program)
        
        # -----------------------
        # get SELECT captures for target nodes that we can mutate
        tree = TS_PARSER.parse(bytes(program, "utf-8"))
        var_rename_captures = get_captures(tree, TS_VARIABLE_DECLARATION_QUERY, "ts","name")
        type_rename_captures = [self.extract_type_from_annotation(n) for n in 
            get_captures(tree, TS_TYPE_ANNOTATIONS_QUERY, "ts","name")]
        remove_annotations_captures = get_captures(tree, TS_PARAM_TYPES_QUERY, "ts","name")
        
        def select_random_subset(x):
            if debug_seed == -1 or len(x) == 0:
                return x
            n = random.randint(1, len(x))
            return random.sample(x, n)
        
        #  random subset of captures
        var_rename_captures = select_random_subset(var_rename_captures)
        type_rename_captures = select_random_subset(type_rename_captures)
        remove_annotations_captures = select_random_subset(remove_annotations_captures)
        
        # -----------------------
        # find ALL ADDITIONAL locations that contain targets
        var_rename_all, type_rename_all, remove_annotations_all = self.find_all_other_locations_of_captures(
            program,
            fim_type,
            var_rename_captures,
            type_rename_captures,
            remove_annotations_captures
        )
        # -----------------------
        # Apply the selected mutations

        new_program, all_mutations = self.mutate_captures(
            program,
            mutations,
            var_rename_all,
            type_rename_all,
            remove_annotations_all
        )
    
        if debug_seed is not None:
            return new_program, all_mutations
        
        return new_program
    
    def find_all_other_locations_of_captures(
        self,
        program:str,
        fim_type:str,
        var_rename_captures: List[tree_sitter.Node],
        type_rename_captures: List[tree_sitter.Node],
        remove_annotations_captures: List[tree_sitter.Node]
    ) -> Tuple[tree_sitter.Node]:
        assert self.tree_sitter_placeholder in program
        types_blacklist = [bytes(fim_type,"utf-8"), bytes(self.tree_sitter_placeholder, "utf-8")]
        tree = TS_PARSER.parse(bytes(program, "utf-8"))
        var_rename_targets = set([x.text for x in var_rename_captures])
        type_rename_targets = set([x.text for x in type_rename_captures])
        
        all_id_captures = get_captures(tree, TS_IDENTIFIER_QUERY, "ts","name")
        all_type_id_captures = get_captures(tree,TS_TYPE_IDENTIFIER_QUERY + TS_PREDEFINED_TYPE_QUERY, "ts","name")
        
        constructor_param_names = set([x.text for x in all_id_captures if self.is_constructor_param(x)])
        var_rename_full_captures = [x for x in all_id_captures 
                                    # rename all ids that match target
                                    if x.text in var_rename_targets
                                    # don't rename constructor params
                                    and not x.text in constructor_param_names
                                    # don't rename built-ins
                                    and not x.text.decode("utf-8") in typescript_builtin_objects
                                    ]
        type_rename_full_captures = [x for x in all_id_captures+all_type_id_captures 
                                    # rename all that match target
                                    if x.text in type_rename_targets
                                    # don't rename forbidden types
                                    and x.text not in types_blacklist
                                    ]

        remove_annotations_captures = [x for x in remove_annotations_captures  if 
                (x.text.replace(b":",b"").strip() != bytes(self.tree_sitter_placeholder, "utf-8"))]

        return var_rename_full_captures, type_rename_full_captures, remove_annotations_captures