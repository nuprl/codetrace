
export enum CompletionItemKind {
	Method = 0,
	Function = 1,
	Constructor = 2,
	Field = 3,
	Variable = 4,
	Class = 5,
	Struct = 6,
	Interface = 7,
	Module = 8,
	Property = 9,
	Event = 10,
	Operator = 11,
	Unit = 12,
	Value = 13,
	Constant = 14,
	Enum = 15,
	EnumMember = 16,
	Keyword = 17,
	Text = 18,
	Color = 19,
	File = 20,
	Reference = 21,
	Customcolor = 22,
	Folder = 23,
	TypeParameter = 24,
	User = 25,
	Issue = 26,
	Snippet = 27
}

export enum CompletionItemInsertTextRule {
	/**
	 * Adjust whitespace/indentation of multiline insert texts to
	 * match the current line indentation.
	 */
	KeepWhitespace = 1,
	/**
	 * `insertText` is a snippet.
	 */
	InsertAsSnippet = 4
}

export interface ISingleEditOperation {
    /**
     * The range to replace. This can be empty to emulate a simple insert.
     */
    range: IRange;
    /**
     * The text to replace with. This can be null to emulate a simple delete.
     */
    text: string | null;
    /**
     * This indicates that this operation has "insert" semantics.
     * i.e. forceMoveMarkers = true => if `range` is collapsed, all markers at the position will be moved.
     */
    forceMoveMarkers?: boolean;
}

export interface IRange {
    /**
     * Line number on which the range starts (starts at 1).
     */
    readonly startLineNumber: number;
    /**
     * Column on which the range starts in line `startLineNumber` (starts at 1).
     */
    readonly startColumn: number;
    /**
     * Line number on which the range ends.
     */
    readonly endLineNumber: number;
    /**
     * Column on which the range ends in line `endLineNumber`.
     */
    readonly endColumn: number;
}
export interface CompletionItemLabel {
    label: string;
    detail?: string;
    description?: string;
}

export enum CompletionItemTag {
    Deprecated = 1
}

export interface UriComponents {
    scheme: string;
    authority: string;
    path: string;
    query: string;
    fragment: <FILL>;
}

export interface IMarkdownString {
    readonly value: string;
    readonly isTrusted?: boolean;
    readonly supportThemeIcons?: boolean;
    readonly supportHtml?: boolean;
    readonly baseUri?: UriComponents;
    uris?: {
        [href: string]: UriComponents;
    };
}

export interface CompletionItemRanges {
    insert: IRange;
    replace: IRange;
}

export interface Command {
    id: string;
    title: string;
    tooltip?: string;
    arguments?: any[];
}

export interface CompletionItem {
    /**
     * The label of this completion item. By default
     * this is also the text that is inserted when selecting
     * this completion.
     */
    label: string | CompletionItemLabel;
    /**
     * The kind of this completion item. Based on the kind
     * an icon is chosen by the editor.
     */
    kind: CompletionItemKind;
    /**
     * A modifier to the `kind` which affect how the item
     * is rendered, e.g. Deprecated is rendered with a strikeout
     */
    tags?: ReadonlyArray<CompletionItemTag>;
    /**
     * A human-readable string with additional information
     * about this item, like type or symbol information.
     */
    detail?: string;
    /**
     * A human-readable string that represents a doc-comment.
     */
    documentation?: string | IMarkdownString;
    /**
     * A string that should be used when comparing this item
     * with other items. When `falsy` the {@link CompletionItem.label label}
     * is used.
     */
    sortText?: string;
    /**
     * A string that should be used when filtering a set of
     * completion items. When `falsy` the {@link CompletionItem.label label}
     * is used.
     */
    filterText?: string;
    /**
     * Select this item when showing. *Note* that only one completion item can be selected and
     * that the editor decides which item that is. The rule is that the *first* item of those
     * that match best is selected.
     */
    preselect?: boolean;
    /**
     * A string or snippet that should be inserted in a document when selecting
     * this completion.
     */
    insertText: string;
    /**
     * Additional rules (as bitmask) that should be applied when inserting
     * this completion.
     */
    insertTextRules?: CompletionItemInsertTextRule;
    /**
     * A range of text that should be replaced by this completion item.
     *
     * Defaults to a range from the start of the {@link TextDocument.getWordRangeAtPosition current word} to the
     * current position.
     *
     * *Note:* The range must be a {@link Range.isSingleLine single line} and it must
     * {@link Range.contains contain} the position at which completion has been {@link CompletionItemProvider.provideCompletionItems requested}.
     */
    range: IRange | CompletionItemRanges;
    /**
     * An optional set of characters that when pressed while this completion is active will accept it first and
     * then type that character. *Note* that all commit characters should have `length=1` and that superfluous
     * characters will be ignored.
     */
    commitCharacters?: string[];
    /**
     * An optional array of additional text edits that are applied when
     * selecting this completion. Edits must not overlap with the main edit
     * nor with themselves.
     */
    additionalTextEdits?: ISingleEditOperation[];
    /**
     * A command that should be run upon acceptance of this item.
     */
    command?: Command;
}

export interface CompletionList {
    suggestions: CompletionItem[];
    incomplete?: boolean;
    dispose?(): void;
}

export type Thenable<T> = PromiseLike<T>;

export type ProviderResult<T> = T | undefined | null | Thenable<T | undefined | null>;


const keyword_map_c = [
	['auto', 'create a automatic variable'],
	['break', 'break out of a loop'],
	['continue', 'continue with the next iteration of a loop'],
	['char', 'character type variable'],
	['const', 'constant variable'],
	['double', 'double precision floating point variable'],
	['float', 'single precision floating point variable'],
	['enum', 'enumeration type'],
	['extern', 'external declaration'],
	['goto', 'jump to a label'],
	['int', 'integer type variable'],
	['long', 'long integer type variable'],
	['signed', 'signed integer type variable'],
	['short', 'short integer type variable'],
	['unsigned', 'unsigned integer type variable'],
	['return', 'return from a function'],
	['sizeof', 'size of a type'],
	['register', 'register variable'],
	['static', 'static variable'],
	['void', 'nothing or no value'],
	['volatile', 'volatile variable (can be changed by hardware)']
];

function generate_completion_keyword([keyword, documentation]: [string, string]) {
	return {
		label: keyword,
		kind: CompletionItemKind.Keyword,
		documentation: documentation,
		insertText: keyword
	};
}

// Snippets Stuff
// for
const for_snippet = ['for (int i = 0; i < 10; i++) {', '    // do something', '}'].join('\n');
// switch-case
const switch_case_snippet = [
	'switch (i) {',
	'    case 1:',
	'        // do something',
	'        break;',
	'    case 2:',
	'        // do something',
	'        break;',
	'    default:',
	'        // do something',
	'        break;',
	'}'
].join('\n');
// do while
const do_while_snippet = ['do {', '    // do something', '} while (i < 10);'].join('\n');
// while
const while_snippet = ['while (i < 10) {', '    // do something', '}'].join('\n');
// if
const if_snippet = ['if (i < 10) {', '    // do something', '}'].join('\n');
// if else
const if_else_snippet = ['if (i < 10) {', '    // do something', '} else {', '    // do something else', '}'].join('\n');
// if else if else
const if_else_if_else_snippet = ['if (i < 10) {', '    // do something', '} else if (i < 20) {', '    // do something else', '} else {', '    // do something else', '}'].join(
	'\n'
);
// enum
const enum_snippet = ['enum {', '    A,', '    B,', '    C', '}'].join('\n');
// struct
const struct_snippet = ['struct {', '    int a;', '    int b;', '}'].join('\n');
// union
const union_snippet = ['union {', '    int a;', '    int b;', '}'].join('\n');

// Special snippets
// standard starter template
const sst_snippet = ['#include <stdio.h>', '#include <stdlib.h>', '', 'int main() {', '    // your code here', '    return 0;', '}'].join('\n');
// main function template
const fn_main_snippet = ['int main() {', '    // your code here', '    return 0;', '}'].join('\n');
// integer function template with 2 parameters
const fn_int_2_snippet = ['int function(int a, int b) {', '    // your code here', '    return 0;', '}'].join('\n');
// printf function useage
const printf_snippet = ['printf("%d", 1);'].join('\n');
// scanf function useage
const scanf_snippet = ['scanf("%d", &i);'].join('\n');
// pointer declaration
const ptr_snippet = ['int *ptr;'].join('\n');

export const snippets = [
	['for', for_snippet, 'for loop'],
	['switch-case', switch_case_snippet, 'switch-case statements with default'],
	['do-while', do_while_snippet, 'do-while loop'],
	['while', while_snippet, 'while loop'],
	['if', if_snippet, 'if statement'],
	['if-else', if_else_snippet, 'if-else statement'],
	['if-else-if-else', if_else_if_else_snippet, 'if (else-if) else statement'],
	['enum', enum_snippet, 'enumeration type'],
	['struct', struct_snippet, 'struct type'],
	['union', union_snippet, 'union type'],
	['sst', sst_snippet, 'standard starter template for C program'],
	['fn-main', fn_main_snippet, 'main function template'],
	['fn-int-2', fn_int_2_snippet, 'integer function template with 2 parameters'],
	['printf', printf_snippet, 'printf function usage'],
	['scanf', scanf_snippet, 'scanf function usage'],
	['ptr', ptr_snippet, 'pointer declaration']
];

function generate_completion_snippet([label, snippet, documentation]: [string, string, string]) {
	return {
		label: label,
		kind: CompletionItemKind.Snippet,
		documentation: documentation,
		insertText: snippet,
		insertTextRules: CompletionItemInsertTextRule.InsertAsSnippet
	};
}

export const completions = [...snippets.map(generate_completion_snippet), ...keyword_map_c.map(generate_completion_keyword)];




function map_to_range(obj, range) {
    obj['range'] = range;
    return obj
}


export const provide_completion_items = (model, position, context, token): ProviderResult<CompletionList> => {
    const word = model.getWordUntilPosition(position);
    const range = {
        startLineNumber: position.lineNumber,
        endLineNumber: position.lineNumber,
        startColumn: word.startColumn,
        endColumn: word.endColumn
    };
    const suggestions = [];
    for (const item of completions) {
        suggestions.push(map_to_range(item, range));
    }
    return {suggestions: [...suggestions]}
}

// Provide all Keywords, snippets for if else, functions etc
