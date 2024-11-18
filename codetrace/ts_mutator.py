import tree_sitter
from codetrace.parsing_utils import replace_between_bytes, get_captures, TS_PARSER, typescript_builtin_objects
import random
from typing import List, Tuple, Union, Callable
from dataclasses import dataclass
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

class TreeSitterLocation:
    start_byte : int
    end_byte : int
    start_point : Tuple[int, int]
    end_point : Tuple[int, int]
    
    def __init__(self, tree_sitter_capture : Union[tree_sitter.Node, Tuple[tree_sitter.Node, str]]):
        if isinstance(tree_sitter_capture, tuple):
            tree_sitter_node, _ = tree_sitter_capture
        else:
            tree_sitter_node = tree_sitter_capture
        self.start_byte = tree_sitter_node.start_byte
        self.end_byte = tree_sitter_node.end_byte
        self.start_point = tree_sitter_node.start_point
        self.end_point = tree_sitter_node.end_point
        
        
    def __repr__(self):
        return f"""TreeSitterLocation(
            start_byte={self.start_byte},
            end_byte={self.end_byte},
            start_point={self.start_point},
            end_point={self.end_point}
    )"""

@dataclass
class Mutation:
    location : TreeSitterLocation
    byte_replacement : bytes
    prefix : Union[bytes, None] = None
    
    def __repr__(self):
        prefix = "None"
        if self.prefix is not None:
            prefix = str(self.prefix)
            
        return f"""Mutation(
                {self.location.__repr__()},
                replacement={str(self.byte_replacement)},
                prefix={prefix}
            )"""

def rename_vars(var_captures : List[Tuple[tree_sitter.Node,str]]) -> List[Mutation]:
    """
    Make mutations for renaming vraiables in VAR_CAPTURES.
    NOTE: new name cannot exist elsewhere in the program, must be different in format from type names.
    The format for vars this function uses is: __tmp{var_index}
    We assume the program does not naturally contain variables with this format
    """
    # map names to captures
    all_names = set([x.text for x in var_captures])
    # map name to new name
    name_to_new_name = {name : bytes(f"__tmp{i}","utf-8") for i, name in enumerate(all_names)}
    mutations = []
    for capture in var_captures:
        location = TreeSitterLocation(capture)
        replacement = name_to_new_name[capture.text]
        mutation = Mutation(location, replacement)
        mutations.append(mutation)
    return mutations

def needs_alias(node : tree_sitter.Node) -> bool:
    """
    Whether the node, when renamed, will need a type alias to be added to the program.
    Includes:
    - predefined types
    - builtin objects
    """
    return node.type == "predefined_type" or node.text.decode("utf-8") in typescript_builtin_objects
    
def rename_types(type_captures : List[Tuple[tree_sitter.Node,str]]) -> List[Mutation]:
    """
    Make mutations for renaming types. Assign a new name to each type in type_captures.
    If a type needs it, we create a new type alias for its renamed version.
    
    NOTE: new name cannot exist elsewhere in the program, must be different in format from variable names.
    We assume the program does not naturally contain types with format __typ{type_index}
    """
    # map names to captures
    all_names = set([x.text for x in type_captures])
    # map names to new names
    name_to_new_name = {name : bytes(f"__typ{i}","utf-8") for i, name in enumerate(all_names)}
    
    mutations = []
    for capture in type_captures:
        location = TreeSitterLocation(capture)
        replacement = name_to_new_name[capture.text]
        
        if needs_alias(capture) and capture.text.decode("utf-8") in typescript_builtin_objects:
                prefix = b"class " + replacement + b" extends " + capture.text + b" {};"
        elif needs_alias(capture):
                prefix = b"type " + replacement + b" = " + capture.text + b";"
        else:
            prefix = None
        mutation = Mutation(location, replacement, prefix)
        mutations.append(mutation)
    return mutations

def delete_annotations(annotation_captures : List[Tuple[tree_sitter.Node,str]])-> List[Mutation]:
    """
    Delete the type annotations from captures
    """
    mutations = []
    for capture in annotation_captures:
        location = TreeSitterLocation(capture)
        mutation = Mutation(location, b"")
        mutations.append(mutation)
    return mutations

def apply_mutations(program : str, mutations : List[Mutation]) -> str:
    """
    Apply mutations to the program.
    NOTE: 
    - applies from bottom up in order to not disturb the byte offsets of other mutations
    - there's the issue that type rename mutations may be nested inside remove annotation mutations
        therefore, if a mutation is nested inside another mutation, keep only the parent mutation
    """
    # take care of nested mutations
    mutations = merge_nested_mutation(mutations)
    mutations.sort(key=lambda x: x.location.start_byte, reverse=True)
    
    byte_program = program.encode("utf-8")
    prefixes = []
    for mutation in mutations:
        byte_program = replace_between_bytes(byte_program, mutation.location.start_byte, mutation.location.end_byte, mutation.byte_replacement)
        if mutation.prefix != None:
            prefixes.append(mutation.prefix)

    byte_program = byte_program.decode("utf-8")
    if len(prefixes) > 0:
        prefixes = "\n".join([p.decode("utf-8") for p in set(prefixes)]) + "\n\n"
        byte_program = prefixes + byte_program
    return byte_program

def merge_nested_mutation(mutations : List[Mutation]) -> List[Mutation]:
    """
    Merge nested annotation mutations. 
    """
    mutations.sort(key=lambda x: x.location.start_byte, reverse=True)
    new_mutations = []
    # work in pairs, if next capture is a superset of the current one, skip curr
    for (curr, prev) in zip(mutations, mutations[1:]):
        if not (curr.location.start_point[0] == prev.location.start_point[0] and 
            curr.location.start_point[1] >= prev.location.start_point[1] and
            curr.location.end_point[0] == prev.location.end_point[0] and
            curr.location.end_point[1] <= prev.location.end_point[1]):
            new_mutations.append(curr)
            
    # add the last capture in all cases
    new_mutations.append(mutations[-1])       
    return new_mutations

def apply_random_mutations_by_kind(program : str, fim_type : str, mutations : List[Callable]) -> str:
    """
    Apply random combination of mutations to the program.
    NOTE: does rename variables first, then rename types, then delete
    """
    new_program = program
    if rename_vars in mutations:
        p = random_mutate(new_program, fim_type, [rename_vars])
        if p != None:
            new_program = p
            
    if rename_types in mutations:
        p = random_mutate(new_program, fim_type, [rename_types])
        if p != None:
            new_program = p
            
    if delete_annotations in mutations:
        p = random_mutate(new_program, fim_type, [delete_annotations])
        if p != None:
            new_program = p
            
    return new_program

def is_constructor_param(x: tree_sitter.Node):
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

def random_mutate(program : str, fim_type : str, mutations : List[Callable], debug_seed : int = None) -> str:
    """
    Apply random combination of mutations to the program.
    Can provide a random seed DEBUG_SEED for debugging.
    NOTE: if debug_seed is -1, this is a special case where we do not select a random subset but
    and run the full set instead (DEGUB only)
    """
    random.seed(debug_seed)
    # to prevent tree-sitter error:
    program = program.replace("<FILL>", "_CodetraceSpecialPlaceholder_")
    # do not rename or delete these types
    types_blacklist = [bytes(fim_type,"utf-8"), bytes("_CodetraceSpecialPlaceholder_", "utf-8")]
    
    # -----------------------
    # get SELECT captures for target nodes that we can mutate
    tree = TS_PARSER.parse(bytes(program, "utf-8"))
    var_rename_captures = get_captures(tree, TS_VARIABLE_DECLARATION_QUERY, "ts","name")
    type_rename_captures = get_captures(tree, TS_TYPE_ANNOTATIONS_QUERY, "ts","name")
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
    var_rename_targets = set([x.text for x in var_rename_captures])
    type_rename_targets = set([x.text.replace(b":",b"").strip() for x in type_rename_captures])
    
    all_id_captures = get_captures(tree, TS_IDENTIFIER_QUERY, "ts","name")
    all_type_id_captures = get_captures(tree,TS_TYPE_IDENTIFIER_QUERY + TS_PREDEFINED_TYPE_QUERY, "ts","name")
    
    constructor_param_names = set([x.text for x in all_id_captures if is_constructor_param(x)])
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
                                   (x.text.replace(b":",b"").strip() != bytes("_CodetraceSpecialPlaceholder_", "utf-8"))]
    
    # -----------------------
    # Apply the selected mutations

    for captures in [var_rename_full_captures, type_rename_full_captures,remove_annotations_captures]:
        # if any out of the selected mutations has no captures, return None
        if len(captures) == 0:
            return None
    
    # collects mutations
    all_mutations = []
    if rename_vars in mutations:
        all_mutations += rename_vars(var_rename_full_captures)
    if rename_types in mutations:
        all_mutations += rename_types(type_rename_full_captures)
    if delete_annotations in mutations:
        all_mutations += delete_annotations(remove_annotations_captures)

    # actually modify the program
    new_program = apply_mutations(program, all_mutations)
    if new_program == program:
        return None
    
    # sometimes the placeholder can be deleted, for example in nested type annotations,
    # so here's a safety check
    if not "_CodetraceSpecialPlaceholder_" in new_program:
        return None
    
    new_program = new_program.replace("_CodetraceSpecialPlaceholder_", "<FILL>")
    
    if debug_seed is not None:
        return new_program, all_mutations
    
    return new_program

def map_random_mutations(iterable, mutations : List[Callable]):
    """
    Apply random combination of mutations
    """
    mutation_names = [m.__name__ for m in mutations]
    new_ds = []
    for _, ex in enumerate(iterable):
        new_program = None
        program, fim_type= ex["fim_program"], ex["fim_type"]
        tries = 0
        while new_program is None and tries < 10:
            tries += 1
            new_program = apply_random_mutations_by_kind(program, fim_type, mutations)

        if new_program != None:
            new_ds.append({"mutated_program": new_program, "mutations" : mutation_names, **ex})
    
    return new_ds