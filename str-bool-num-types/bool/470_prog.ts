/**
 * Number
 */
type NumberField = {
  type: 'number';
  canBeNull: boolean;
  default: number | null;
  primaryKey: boolean;
};

export const number = (options: {
  canBeNull?: boolean;
  default?: number | null;
  primaryKey?: boolean;
}): NumberField => {
  return {
    type: 'number',
    canBeNull: options.canBeNull || false,
    default: options.default || null,
    primaryKey: options.primaryKey || false,
  };
};

/**
 * BigInt
 */
type BigIntField = {
  type: 'bigint';
  canBeNull: boolean;
  default: number | null;
  primaryKey: boolean;
};

export const bigInt = (options: {
  canBeNull?: boolean;
  default?: number | null;
  primaryKey?: boolean;
}): BigIntField => {
  return {
    type: 'bigint',
    canBeNull: options.canBeNull || false,
    default: options.default || null,
    primaryKey: options.primaryKey || false,
  };
};

/**
 * String
 */
type StringField = {
  type: 'string';
  maxLength: number;
  canBeNull: boolean;
  default: string | null;
  primaryKey: boolean;
};

export const string = (options: {
  maxLength: number;
  canBeNull?: boolean;
  default?: string | null;
  primaryKey?: boolean;
}): StringField => {
  return {
    type: 'string',
    maxLength: options.maxLength,
    canBeNull: options.canBeNull || false,
    default: options.default || null,
    primaryKey: options.primaryKey || false,
  };
};

/**
 * Boolean
 */
type BooleanField = {
  type: 'boolean';
  canBeNull: boolean;
  default: boolean | null;
  primaryKey: <FILL>;
};

export const boolean = (options: {
  canBeNull?: boolean;
  default?: boolean | null;
  primaryKey?: boolean;
}): BooleanField => {
  return {
    type: 'boolean',
    canBeNull: options.canBeNull || false,
    default: options.default || null,
    primaryKey: options.primaryKey || false,
  };
};

/**
 * Enumerated
 */
type EnumeratedField = {
  type: 'enumerated';
  values: string[];
  canBeNull: boolean;
  default: string | null;
  primaryKey: boolean;
};

export const enumerated = (options: {
  values: string[];
  canBeNull?: boolean;
  default?: string | null;
  primaryKey?: boolean;
}): EnumeratedField => {
  return {
    type: 'enumerated',
    values: options.values,
    canBeNull: options.canBeNull || false,
    default: options.default || null,
    primaryKey: options.primaryKey || false,
  };
};

/**
 * DateTime
 */
type DateTimeField = {
  type: 'datetime';
  canBeNull: boolean;
  default: Date | null;
  primaryKey: boolean;
};

export const dateTime = (options: {
  canBeNull?: boolean;
  default?: Date | null;
  primaryKey?: boolean;
}): DateTimeField => {
  return {
    type: 'datetime',
    canBeNull: options.canBeNull || false,
    default: options.default || null,
    primaryKey: options.primaryKey || false,
  };
};

/**
 * Any Field
 */
export type AnyField =
  | NumberField
  | BigIntField
  | StringField
  | BooleanField
  | EnumeratedField
  | DateTimeField;

export type Fields = {
  [key: string]: AnyField;
};
