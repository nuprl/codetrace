export class ParseError extends Error {
  value: unknown;
  config: ParserConfig<any>;
  reason?: unknown;

  constructor(message: <FILL>, value: unknown, config: ParserConfig<any>, reason?: unknown) {
    let errorMessage = message;
    if (reason != null) {
      errorMessage += `\nReason:\n${String(reason)}`;
    }
    errorMessage += `\nValue:\n${JSON.stringify(value, null, 2)}`;
    super(errorMessage);
    this.value = value;
    this.config = config;
    this.reason = reason;
  }
}

export interface ParserConfig<T, TPreTransform = T> {
  /**
   * Parses the type using either a primitive value compatible with typeof
   * or a type guard function.
   */
  type: string | ((value: unknown) => value is TPreTransform);
  /** If set to true, null or undefined values will be parsed without error. */
  optional?: true;
  /**
   * Parser or a record of parsers for the enumerable fields of an object.
   *
   * If passed a record of parsers, a strict set of fields will be parsed using
   * a specific parse config for each field.
   *
   * If passed a single parser, an arbitrary set of fields will be parsed using
   * the same parse config for each field.
   *
   * If the value is not an object, this config field is ignored.
   */
  fields?:
    | {
        [K in keyof TPreTransform]: Parser<TPreTransform[K], any>;
      }
    | Parser<TPreTransform[keyof TPreTransform], any>;
  /**
   * Parser for the elements of a list.
   *
   * If the value is not an array, this config field is ignored.
   */
  elements?: TPreTransform extends unknown[] ? Parser<TPreTransform[number], any> : never;
  /**
   * Transforms the value when parsing using the provided function.
   *
   * If field or element parse configs include transforms, those transforms
   * will be applied prior to the overall transform function.
   */
  transform?: (value: TPreTransform) => T;
}

export class Parser<T, TPreTransform = T> {
  config: ParserConfig<T, TPreTransform>;

  constructor(
    config:
      | ParserConfig<T, TPreTransform>
      | ((self: Parser<T, TPreTransform>) => ParserConfig<T, TPreTransform>)
  ) {
    this.config = typeof config === "function" ? config(this) : config;
  }

  /**
   * Parses a value using the provided config, and throws a ParseError if the
   * value could not be parsed.
   */
  parse(value: unknown): T {
    const { optional, type, fields, elements, transform } = this.config;

    if (value == null) {
      if (optional != true) {
        throw new ParseError("Required value was null, undefined, or unset", value, this.config);
      } else {
        return value as any as T;
      }
    }

    if (typeof type === "string" && typeof value !== type) {
      throw new ParseError("Value was not the expected type", value, this.config);
    }

    if (typeof type === "function" && !type(value)) {
      throw new ParseError(
        "Value was not the expected type; value did not pass type guard",
        value,
        this.config
      );
    }

    let parsedPreTransform: any;

    if (elements && Array.isArray(value)) {
      parsedPreTransform = value.map((element) => elements.parse(element));
    } else if (fields && typeof value === "object" && !Array.isArray(value)) {
      parsedPreTransform = {};
      if (fields instanceof Parser) {
        for (const key in value) {
          parsedPreTransform[key] = fields.parse((value as any)[key]);
        }
      } else {
        for (const key in fields) {
          parsedPreTransform[key] = fields[key].parse((value as any)[key]);
        }
        for (const key in value) {
          if (!(key in fields)) {
            throw new ParseError("Unexpected field in value", value, this.config);
          }
        }
      }
    } else {
      parsedPreTransform = value;
    }

    if (transform) {
      try {
        return transform(parsedPreTransform as TPreTransform);
      } catch (transformErr) {
        throw new ParseError(
          "Could not transform value while parsing",
          parsedPreTransform,
          this.config,
          transformErr
        );
      }
    }

    return parsedPreTransform as T;
  }
}

export const stringParser = new Parser<string>({ type: "string" });
export const numberParser = new Parser<number>({ type: "number" });
export const booleanParser = new Parser<boolean>({ type: "boolean" });
export const unknownRecordParser = new Parser<Record<string, unknown>>({
  type: (value: unknown): value is Record<string, unknown> =>
    typeof value === "object" &&
    value != null &&
    Object.keys(value).every((key) => typeof key === "string"),
});
export const dateStringParser = new Parser<Date, string>({
  type: "string",
  transform(dateString) {
    const parsedDate = new Date(dateString);
    if (parsedDate.toString() === "Invalid Date") {
      throw new Error("Invalid date string");
    }
    return parsedDate;
  },
});
export function makeEnumParser<TEnumType, TEnumObj extends Record<string, any>>(enumObj: TEnumObj) {
  return new Parser<TEnumType>({
    type: (value: unknown): value is TEnumType => Object.values(enumObj).includes(value),
  });
}
