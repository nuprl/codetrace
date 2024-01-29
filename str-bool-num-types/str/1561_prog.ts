export const Kind = 'Core#WebApiIndex';

/**
 * @deprecated This was used in the old version of ARC.
 */
export interface ILegacyWebApiIndex {
  /**
   * API title
   */
  title: string;
  /**
   * API media type
   * @deprecated This has been renamed to `vendor`.
   */
  type: string;
  /**
   * API order on the list
   */
  order: number;
  /**
   * List of version names stored with this API.
   */
  versions: string[];
  /**
   * The latest added version name.
   */
  latest: string;
}

export interface IWebApiIndex {
  kind: typeof Kind;
  /**
   * API title
   */
  title: string;
  /**
   * List of version names stored with this API.
   */
  versions: string[];
  /**
   * The latest added version name.
   */
  latest: string;
  /**
   * The API vendor. E.g. RAML 1.0, OAS 3.0, ASYNC 2.0, ...
   */
  vendor: <FILL>;
}

export class WebApiIndex {
  kind = Kind;
  /**
   * API title
   */
  title = '';
  /**
   * List of version names stored with this API.
   */
  versions: string[] = [];
  /**
   * The latest added version name.
   */
  latest = '';
  /**
   * The API vendor. E.g. RAML 1.0, OAS 3.0, ASYNC 2.0, ...
   */
  vendor = '';

  static isLegacy(api: unknown): boolean {
    const legacy = api as ILegacyWebApiIndex;
    return !!legacy.type;
  }

  static fromLegacy(api: ILegacyWebApiIndex): WebApiIndex {
    const { title, type, versions=[], latest } = api;
    const init: IWebApiIndex = {
      kind: Kind,
      title,
      versions,
      latest,
      vendor: type,
    };
    return new WebApiIndex(init);
  }

  constructor(input?: string|IWebApiIndex) {
    let init: IWebApiIndex;
    if (typeof input === 'string') {
      init = JSON.parse(input);
    } else if (typeof input === 'object') {
      init = input;
    } else {
      init = {
        kind: Kind,
        latest: '',
        title: '',
        vendor: '',
        versions: [],
      };
    }
    this.new(init);
  }

  new(init: IWebApiIndex): void {
    const { latest='', title='', vendor='', versions=[] } = init;
    this.latest = latest;
    this.versions = versions;
    this.title = title;
    this.vendor = vendor;
  }

  toJSON(): IWebApiIndex {
    const result: IWebApiIndex = {
      kind: Kind,
      latest: this.latest,
      versions: this.versions,
      title: this.title,
      vendor: this.vendor,
    };
    return result;
  }
}
