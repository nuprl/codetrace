export interface IUrlValueParserOptions {
  /**
   * A query string delimiter to use when processing query parameters.
   */
  queryDelimiter?: string;
}

interface DataValues {
  /**
   * A protocol value in format `protocol` + ':'
   */
  protocol?: string;
  /**
   * The authority part of the URL value
   */
  host?: string;
  /**
   * Path part of the URL.
   */
  path?: string;
  /**
   * Anchor part of the URL.
   */
  anchor?: <FILL>;
  /**
   * Search part of the URL.
   */
  search?: string;
  opts?: IUrlValueParserOptions | undefined;
}

/**
 * Implements logic for parsing URL string.
 */
export class UrlValueParser {
  protected __data: DataValues;

  constructor(opts?: IUrlValueParserOptions) {
    this.__data = {};
    this.opts = opts;
  }

  /**
   * @returns Class options.
   */
  get opts(): IUrlValueParserOptions {
    return this.__data.opts || {
      queryDelimiter: '&',
    };
  }

  /**
   * Sets parser options.
   * Unknown options are ignored.
   */
  set opts(opts: IUrlValueParserOptions | undefined) {
    const options = (opts || {}) as IUrlValueParserOptions;
    this.__data.opts = {
      queryDelimiter: options.queryDelimiter || '&'
    };
  }

  /**
   * Returns protocol value in format `protocol` + ':'
   *
   * @param value URL to parse.
   * @return Value of the protocol or undefined if value not set
   */
  protected _parseProtocol(value: string): string | undefined {
    if (!value) {
      return undefined;
    }
    const delimiterIndex = value.indexOf('://');
    if (delimiterIndex !== -1) {
      return value.substring(0, delimiterIndex + 1);
    }
    return undefined;
  }

  /**
   * Gets a host value from the url.
   * It reads the whole authority value of given `value`. It doesn't parses it
   * to host, port and
   * credentials parts. For URL panel it's enough.
   *
   * @param value The URL to parse
   * @return Value of the host or undefined.
   */
  protected _parseHost(value: string): string | undefined {
    if (!value) {
      return undefined;
    }
    let result = value;
    const delimiterIndex = result.indexOf('://');
    if (delimiterIndex !== -1) {
      result = result.substring(delimiterIndex + 3);
    }
    if (!result) {
      return undefined;
    }
    // We don't need specifics here (username, password, port)
    const host = result.split('/')[0];
    return host;
  }

  /**
   * Parses the path part of the URL.
   *
   * @param value URL value
   * @returns Path part of the URL
   */
  protected _parsePath(value: string): string | undefined {
    if (!value) {
      return undefined;
    }
    let result = value;
    const isBasePath = result[0] === '/';
    if (!isBasePath) {
      const index = result.indexOf('://');
      if (index !== -1) {
        result = result.substring(index + 3);
      }
    }
    let index = result.indexOf('?');
    if (index !== -1) {
      result = result.substring(0, index);
    }
    index = result.indexOf('#');
    if (index !== -1) {
      result = result.substring(0, index);
    }
    const lastIsSlash = result[result.length - 1] === '/';
    const parts = result.split('/').filter((part) => !!part);
    if (!isBasePath) {
      parts.shift();
    }
    let path = `/${  parts.join('/')}`;
    if (lastIsSlash && parts.length > 1) {
      path += '/';
    }
    return path;
  }

  /**
   * Returns query parameters string (without the '?' sign) as a whole.
   *
   * @param value The URL to parse
   * @returns Value of the search string or undefined.
   */
  protected _parseSearch(value: string): string | undefined {
    if (!value) {
      return undefined;
    }
    let index = value.indexOf('?');
    if (index === -1) {
      return undefined;
    }
    const result = value.substring(index + 1);
    index = result.indexOf('#');
    if (index === -1) {
      return result;
    }
    return result.substring(0, index);
  }

  /**
   * Reads a value of the anchor (or hash) parameter without the `#` sign.
   *
   * @param value The URL to parse
   * @returns Value of the anchor (hash) or undefined.
   */
  protected _parseAnchor(value: string): string | undefined {
    if (!value) {
      return undefined;
    }
    const index = value.indexOf('#');
    if (index === -1) {
      return undefined;
    }
    return value.substring(index + 1);
  }

  /**
   * Returns an array of items where each item is an array where first
   * item is param name and second is it's value. Both always strings.
   *
   * @param search Parsed search parameter
   * @returns Always returns an array.
   */
  protected _parseSearchParams(search?: string): string[][] {
    const result: string[][] = [];
    if (!search) {
      return result;
    }
    const parts = search.split(this.opts.queryDelimiter as string);
    parts.forEach((item) => {
      const _part = ['', ''];
      const _params = item.split('=');
      let _name = _params.shift();
      if (!_name && _name !== '') {
        return;
      }
      _name = _name.trim();
      const _value = _params.join('=').trim();
      _part[0] = _name;
      _part[1] = _value;
      result.push(_part);
    });
    return result;
  }
}
