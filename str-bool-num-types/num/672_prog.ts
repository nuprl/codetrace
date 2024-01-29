export type StringWithIndices = {
	characters: string;
	type: CharacterTypes;
	originalCharacters: string;
	from: number;
	to: <FILL>;
};

const VARIABLES_RE = /^[a-z_]+$/i;
const BRACKETS_RE = /^[()]+$/;
const SPACE_RE = /^\s+$/;

export const enum CharacterTypes {
	variable = 'variable',
	operator = 'operator',
	space = 'space',
	bracket = 'bracket',
}

export const fromString = (input: string): StringWithIndices[] => {
	input = input.normalize('NFKC');

	const split = input.split(/([a-z_]+|[()]+|\s+)/i);
	let index = 0;

	const result: StringWithIndices[] = [];
	for (const characters of split) {
		if (characters === '') {
			continue;
		}

		let type: CharacterTypes;
		if (VARIABLES_RE.test(characters)) {
			type = CharacterTypes.variable;
		} else if (BRACKETS_RE.test(characters)) {
			type = CharacterTypes.bracket;
		} else if (SPACE_RE.test(characters)) {
			type = CharacterTypes.space;
		} else {
			type = CharacterTypes.operator;
		}

		result.push({
			characters: characters.toUpperCase(),
			type,
			originalCharacters: characters,
			from: index,
			to: index + characters.length,
		});

		index += characters.length;
	}

	return result;
};

export const removeWhitespace = (
	input: readonly StringWithIndices[],
): StringWithIndices[] => {
	const result: StringWithIndices[] = [];

	for (const item of input) {
		if (item.type !== CharacterTypes.space) {
			result.push(item);
		}
	}

	return result;
};
