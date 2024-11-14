import tree_sitter
from codetrace.parsing_utils import replace_between_bytes, get_captures, PY_PARSER, is_in_capture_range
import random
from typing import List, Tuple, Union, Callable, Generator
from dataclasses import dataclass
import random
import typing
import builtins
from collections import namedtuple
"""
https://github.com/nvim-treesitter/nvim-treesitter/blob/master/queries/python/highlights.scm
Random mutation code.

Some considerations.

1. renaming to an arbitrary name (especially length)

the trick to being able to rename to any name is to accumulate
all the changes and apply them finally from end to start

2. each mutation method should produce different types of names to prevent overlap
and also for semantics
"""

# There are ~114 types of non-terminals listed in the Python tree-sitter
# grammar:
#
# https://github.com/tree-sitter/tree-sitter-python/blob/master/src/node-types.json
#
# These are the statements that cannot contain references to variables bound
# by function definitions and lambdas. Of course, there are some truly wierd
# cases, for example:
#
#     def foo(c):
#         class c:
#             pass
#
# Moreover, code can dynamically modify the set of names in any scope. But,
# it should be safe to ignore these cases for most code.
IMPORT_STATEMENTS = [
    "import_statement",
    "import_from_statement",
    "import_prefix",
    "future_import_statement",
    "wildcard_import",
    "relative_import"
]
IMPORT_STATEMENT_QUERY = """
((import_statement) @import_statement)
((import_from_statement) @import_statement)
((import_prefix) @import_statement)
((future_import_statement) @import_statement)
((wildcard_import) @import_statement)
((relative_import) @import_statement)
"""

# # There are several other contexts where variables can appear. But, we are being
# # safely lazy. It should be enough to check that we are in one the contexts
# # below and not in the NONVAR_STATEMENTS contexts.
# VAR_CONTEXTS = [
#     "parameters",
#     "module",  # Top-level variable I believe
#     "function_definition",
# ]

# TYPED_IDENTIFIERS = PY_LANGUAGE.query("""(typed_parameter) @param""")


CLASS_NAMES = """(class_definition name: (identifier) @id)"""

####

PY_IDENTIFIER_QUERY = """((identifier) @id)""" # this also captures types
PY_TYPE_IDENTIFIER_QUERY = """[
  (typed_parameter type:
      (type (identifier) @id))
]"""
# This query finds the parameters of function definitions. It does not do
# lambdas.
FUNCTION_PARAMS = """
    [
        (function_definition parameters: 
            (parameters [ (identifier) @id (typed_parameter (identifier) @id) ]))

    ]
"""
FUNCTION_NAME = """(function_definition name: ((identifier) @id))"""
PY_VARIABLE_DECLARATION_QUERY = FUNCTION_PARAMS + FUNCTION_NAME

PY_ATTRIBUTE_IDENTIFIER_QUERY = """(attribute attribute: (identifier) @id)"""

#NOTE: the following captures include colon and identifier (if present)
# eg. cap(n : int) = n : int, cap(-> n) = -> n
# needs postprocessing
PY_TYPE_ANNOTATIONS_QUERY = """((typed_parameter) @annotation)"""

RETURN_TYPES = """  [
        (function_definition return_type: 
            (type (identifier) @id))

]"""


class TreeSitterLocation:
    start_byte : int
    end_byte : int
    start_point : Tuple[int, int]
    end_point : Tuple[int, int]
    
    def __init__(self, tree_sitter_node : tree_sitter.Node):

        # print(tree_sitter_capture, type(tree_sitter_capture))
        # if isinstance(tree_sitter_capture, tuple):
        #     tree_sitter_node, _ = tree_sitter_capture
        # else:
        # tree_sitter_node = tree_sitter_capture
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
        if self.prefix is not None:
            prefix = str(self.prefix)
        else:
            prefix = "None"
            
        return f"""Mutation(
        {self.location.__repr__()},
        replacement={str(self.byte_replacement)},
        prefix={prefix}
    )"""

def rename_vars(var_captures : List[Tuple[tree_sitter.Node,str]], **kwargs) -> List[Mutation]:
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

def needs_alias(typ: bytes, import_statements : bytes) -> bool:
    # if type is a builtin or typing, needs alias
    # if a type is in imports, needs alias
    return any([typ==bytes(t,"utf-8") for t in dir(builtins)+dir(typing)]) or typ in import_statements
    
def rename_types(type_captures : List[Tuple[tree_sitter.Node,str]], **kwargs) -> List[Mutation]:
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
    do_not_prefix = set()
    for capture in type_captures:
        location = TreeSitterLocation(capture)
        replacement = name_to_new_name[capture.text]
        
        if needs_alias(capture.text, kwargs["import_statement_names"]):
            # make new type alias
            prefix = replacement + b" : TypeAlias = " + capture.text
        else:
            prefix = None
        mutation = Mutation(location, replacement, prefix)
        mutations.append(mutation)
    return mutations

def delete_annotations(annotation_captures : List[Tuple[tree_sitter.Node,str]], **kwargs)-> List[Mutation]:
    """
    Delete the type annotations from captures
    """
    mutations = []
    for capture in annotation_captures:
        location = TreeSitterLocation(capture)
        mutation = Mutation(location, b"")
        mutations.append(mutation)
    return mutations

def add_type_aliases_after_imports(code: bytes, type_aliases : List[bytes]) -> bytes:
    """
    Add type aliases to the prefix after the last import statement
    NOTE:we assume all imports are at the top of the file
    """
    import_typ_alias = b"from typing import TypeAlias\n"
    type_aliases = import_typ_alias + b"\n".join(type_aliases) + b"\n\n"
    captures = get_captures(code, IMPORT_STATEMENT_QUERY, "py", "import_statement")
    if len(captures) == 0:
        return type_aliases + code
    # find the last import statement
    last_import = max(captures, key=lambda x: x.end_byte)
    new_code = code[:last_import.end_byte] + b"\n" + type_aliases + code[last_import.end_byte:]
    return new_code

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
        if mutation.prefix is not None:
            prefixes.append(mutation.prefix)

    prefixes = list(set(prefixes))
    if len(prefixes) > 0:
        return add_type_aliases_after_imports(byte_program, prefixes).decode("utf-8")
    else:
        return byte_program.decode("utf-8")

def merge_nested_mutation(mutations : List[Mutation]) -> List[Mutation]:
    """
    Merge nested annotation mutations. 
    """
    mutations.sort(key=lambda x: x.location.start_byte, reverse=True)
    new_mutations = []
    # work in pairs, if next capture is a superset of the current one, skip curr
    for (curr, prev) in zip(mutations, mutations[1:]):
        if (curr.location.start_point[0] == prev.location.start_point[0] and 
            curr.location.start_point[1] >= prev.location.start_point[1] and
            curr.location.end_point[0] == prev.location.end_point[0] and
            curr.location.end_point[1] <= prev.location.end_point[1]):
            continue
        else:
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

def postprocess_type_annotation(node_capture : tree_sitter.Node, target_char : bytes, shift_amt : int) -> tree_sitter.Node:
    """
    Postprocess the annotation node by applying a shift to the node from the target character. 
    Captured annotations contain var id and type id, for example:
        n : int
    We want to extract only:
        : int
    Thus, need to shift node location + text
    """
    text = node_capture.text
    # find the index of the colon
    index = text.index(target_char)
    # count num bytes to shift
    shift = index + shift_amt
    # shift the node
    new_start_byte = node_capture.start_byte + shift
    new_start_point = (node_capture.start_point[0], node_capture.start_point[1] + shift)
    new_text = text[shift:]
    
    # edit node
    TSNodeAlias = namedtuple("TSNodeAlias", ["start_byte", "end_byte", "start_point", "end_point", "text"])
    new_node = TSNodeAlias(new_start_byte, node_capture.end_byte, new_start_point, node_capture.end_point, new_text)
    node_capture = new_node
    assert node_capture.text == new_text, f"Text mismatch: {node_capture.text} != {new_text}"
    return node_capture

def postprocess_return_type(node_capture : tree_sitter.Node, byte_program : bytes) -> tree_sitter.Node:
    """
    Return types in tree sitter don't include the ->, so we need to add it back
    """
    text = node_capture.text
    # find the first index of '->' starting from the end
    index = byte_program[:node_capture.start_byte].rfind(b"->")
    
    new_start_byte = index
    shift = index - node_capture.start_byte
    new_start_point = (node_capture.start_point[0], node_capture.start_point[1] + shift)
    new_text = byte_program[index:node_capture.end_byte]
    
    # edit node
    TSNodeAlias = namedtuple("TSNodeAlias", ["start_byte", "end_byte", "start_point", "end_point", "text"])
    new_node = TSNodeAlias(new_start_byte, node_capture.end_byte, new_start_point, node_capture.end_point, new_text)
    node_capture = new_node
    assert node_capture.text == new_text, f"Text mismatch: {node_capture.text} != {new_text}"
    return node_capture

def random_mutate(program : str, fim_type : str, mutations : List[Callable], debug_seed : int = None) -> str:
    """
    Apply random combination of mutations to the program.
    Can provide a random seed DEBUG_SEED for debugging.
    NOTE: if debug_seed is -1, this is a special case where we do not select a random subset but
    and run the full set instead (DEGUB only)
    """
    if debug_seed is not None:
        random.seed(debug_seed)
        
    # to prevent tree-sitter error:
    program = program.replace("<FILL>", "_CodetraceSpecialPlaceholder_")
    
    # -----------------------
    # get SELECT captures for target nodes that we can mutate
    program_bytes = bytes(program, "utf-8")
    tree = PY_PARSER.parse(program_bytes)

    var_rename_captures = get_captures(tree, PY_VARIABLE_DECLARATION_QUERY, "py", "id")
    return_types_captures = get_captures(tree, RETURN_TYPES, "py", "id")
    class_names = get_captures(tree, CLASS_NAMES, "py", "id")
    type_annotations_captures = get_captures(tree, PY_TYPE_ANNOTATIONS_QUERY, "py", "annotation")
    
    type_rename_captures = [postprocess_type_annotation(x, b":", 1) for x in type_annotations_captures] + class_names + return_types_captures
    remove_annotations_captures = [postprocess_type_annotation(x, b":", 0) for x in type_annotations_captures] 
    remove_annotations_captures +=  [postprocess_return_type(x, program_bytes) for x in return_types_captures]
    
    def select_random_subset(x):
        if debug_seed == -1 or len(x) == 0:
            return x
        n = random.randint(1, len(x))
        return random.sample(x, n)
    
    #  random subset of captures
    var_rename = select_random_subset(var_rename_captures)
    type_rename = select_random_subset(type_rename_captures)
    remove_annotations = select_random_subset(remove_annotations_captures)
    
    # -----------------------
    # find ALL ADDITIONAL locations that contain targets
    
    var_rename_all, type_rename_all, remove_annotations_all = find_mutation_locations(
        program,
        fim_type,
        var_rename,
        type_rename,
        remove_annotations
    )
    
    # -----------------------
    # Apply random combinations of mutations
    
    new_program, all_mutations = mutate_captures(
        program,
        mutations,
        var_rename_all,
        type_rename_all,
        remove_annotations_all
    )
    
    if debug_seed is not None:
        return new_program, all_mutations
    
    return new_program

def all_combinations(*iterables)->Generator:
    for iter in iterables:
        random.shuffle(iter)
    acc = {i:[] for i in range(len(iterables))}
    while any([len(iter) > 0 for iter in iterables]):
        for _id,iter in enumerate(iterables):
            if len(iter) > 0:
                acc[_id].append(iter.pop())
                yield [acc[i] for i in range(len(iterables))]

def incremental_mutate(
    program : str, 
    fim_type : str,
    mutations : List[Callable]
) -> Generator:
    """
    Apply incremental combination of random mutations to the program.
    """
    # to prevent tree-sitter error:
    program = program.replace("<FILL>", "_CodetraceSpecialPlaceholder_")
    
    # -----------------------
    # get SELECT captures for target nodes that we can mutate
    program_bytes = bytes(program, "utf-8")
    tree = PY_PARSER.parse(program_bytes)

    var_rename_captures = get_captures(tree, PY_VARIABLE_DECLARATION_QUERY, "py", "id")
    return_types_captures = get_captures(tree, RETURN_TYPES, "py", "id")
    class_names = get_captures(tree, CLASS_NAMES, "py", "id")
    type_annotations_captures = get_captures(tree, PY_TYPE_ANNOTATIONS_QUERY, "py", "annotation")
    
    type_rename_captures = [postprocess_type_annotation(x, b":", 1) for x in type_annotations_captures] + class_names + return_types_captures
    remove_annotations_captures = [postprocess_type_annotation(x, b":", 0) for x in type_annotations_captures] 
    remove_annotations_captures +=  [postprocess_return_type(x, program_bytes) for x in return_types_captures]

    all_combos = all_combinations(
        var_rename_captures,
        type_rename_captures,
        remove_annotations_captures,
    )

    for (var_rename, type_rename, remove_annotations) in all_combos:
        # -----------------------
        # find ALL ADDITIONAL locations that contain targets
        
        var_rename_all, type_rename_all, remove_annotations_all = find_mutation_locations(
            program,
            fim_type,
            var_rename,
            type_rename,
            remove_annotations
        )
        
        # -----------------------
        # Apply random combinations of mutations
        
        new_program, _ = mutate_captures(
            program,
            mutations,
            var_rename_all,
            type_rename_all,
            remove_annotations_all
        )
        yield new_program

def find_mutation_locations(program:str,
    fim_type:str,
    var_rename_captures: List[tree_sitter.Node],
    type_rename_captures: List[tree_sitter.Node],
    remove_annotations_captures: List[tree_sitter.Node]
) -> Tuple[tree_sitter.Node]:
    var_rename_targets = set([x.text for x in var_rename_captures])
    type_rename_targets = set([x.text for x in type_rename_captures])
    
    # do not rename or delete these types
    types_blacklist = [bytes(fim_type,"utf-8"), 
                       bytes("_CodetraceSpecialPlaceholder_", "utf-8")]
    import_statements = get_captures(program, IMPORT_STATEMENT_QUERY, "py", "import_statement")
    all_id_captures = get_captures(program, PY_IDENTIFIER_QUERY, "py", "id")
    all_attribute_ids = get_captures(program, PY_ATTRIBUTE_IDENTIFIER_QUERY, "py", "id")
    attribute_names = set([x.text for x in all_attribute_ids])
    import_statement_names = b"\n".join([x.text for x in import_statements])
    var_rename_full_captures = [
        x for x in all_id_captures 
        # rename all ids that match target
        if x.text in var_rename_targets
        # don't rename attributes
        and not x.text in attribute_names #TODO: do we want to rename attributes?
        # don't rename built-ins because no alias supported for vars
        and not x.text.decode("utf-8") in dir(builtins)+dir(typing)
        # don't rename anything in import statements because no alias supported for vars
        and not x.text in import_statement_names
    ]
    type_rename_full_captures = [
        x for x in all_id_captures
        # rename all that match target
        if x.text in type_rename_targets
        # don't rename attributes
        and not x.text in attribute_names #TODO: do we want to rename attributes?
        # don't rename forbidden types
        and x.text not in types_blacklist
        # don't rename if in range of import statements 
        # NOTE: we can rename text in import statements because of alias support, but not the actual imports
        and not is_in_capture_range(x, import_statements)
    ]
    remove_annotations_captures = [
        x for x in remove_annotations_captures  
            if (x.text.replace(b":",b"").replace(b"->",b"").strip() != bytes("_CodetraceSpecialPlaceholder_", "utf-8"))
    ]
    return var_rename_full_captures, type_rename_full_captures, remove_annotations_captures

def mutate_captures(
    program:str,
    mutations:List[callable],
    var_rename_captures: List[tree_sitter.Node],
    type_rename_captures: List[tree_sitter.Node],
    remove_captures: List[tree_sitter.Node],
)-> Tuple[str, List[callable]]:
    import_statements = get_captures(program, IMPORT_STATEMENT_QUERY, "py", "import_statement")
    import_statement_names = b"\n".join([x.text for x in import_statements])
    if any([len(captures) == 0 for captures in [var_rename_captures, type_rename_captures,remove_captures]]):
        # if any out of the selected mutations has no captures, return None
        return None,[]
    
    # collects mutations
    all_mutations = []
    if rename_vars in mutations:
        all_mutations += rename_vars(var_rename_captures)
    if rename_types in mutations:
        all_mutations += rename_types(type_rename_captures, import_statement_names=import_statement_names)
    if delete_annotations in mutations:
        all_mutations += delete_annotations(remove_captures)

    # actually modify the program
    new_program = apply_mutations(program, all_mutations)
    if new_program == program:
        # no mods applied, return None
        return None, []
    
    # sometimes the placeholder can be deleted, for example in nested type annotations,
    # so here's a safety check
    if not "_CodetraceSpecialPlaceholder_" in new_program:
        return None, []
    
    new_program = new_program.replace("_CodetraceSpecialPlaceholder_", "<FILL>")
    return new_program, all_mutations

"""
Maps
"""

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
