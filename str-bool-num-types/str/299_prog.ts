export interface LanguageInterface {
  name: string;
  code: string;
  nativeName: string;
  layout: "ltr" | "rtl";
}

export interface LanguagesInterface {
  [key: string]: LanguageInterface;
}

export interface TranslatorOptions {
  cache?: boolean;
  defaultLanguage: string;
  currentLangauge: string;
  engine: TranslateEngine;
}

export type TranslationResponseShape = Record<string, string>;

export type TranslateEngine = (
  language: {
    default: string;
    current: string;
  },
  targets: string[]
) => Promise<TranslationResponseShape> | TranslationResponseShape;

export type CacheSetter = (key: <FILL>, value: string) => Promise<void> | void;
export type CacheGetter = (
  key: string
) => Promise<string | null> | string | null;

export class Translator {
  protected TranslationTargets: string[] = [];
  protected TranslationResults: Record<string, string> = {};
  protected CacheEngines?: {
    setter: CacheSetter;
    getter: CacheGetter;
  };

  // Global Metadata
  public metadata: Record<string, any> = {};

  constructor(protected Options: TranslatorOptions) {
    // Validate Options
    this.setOptions(this.Options);
  }

  public setTranslations(
    translations: string[] | ((translations: string[]) => string[])
  ) {
    if (typeof translations === "object")
      this.TranslationTargets = translations;
    else if (typeof translations === "function") {
      this.TranslationTargets = translations(this.TranslationTargets);
    }

    return this;
  }

  public getTranslated() {
    return this.TranslationResults;
  }

  public setCacheEngines(setter: CacheSetter, getter: CacheGetter) {
    if (typeof setter !== "function")
      throw new Error(`Please provide a valid Cache Setter Function!`);
    else if (typeof getter !== "function")
      throw new Error(`Please provide a valid Cache Getter Function!`);

    this.CacheEngines = { setter, getter };
    return this;
  }

  public setOptions(options: Partial<TranslatorOptions>) {
    // Validate Options
    if (options.engine && typeof options.engine !== "function")
      throw new Error(`Please provide a valid Translation Engine Function!`);

    // Set Options
    this.Options = {
      ...this.Options,
      ...options,
    };

    return this;
  }

  public async setCache(key: string, value: string) {
    if (
      [undefined, true].includes(this.Options.cache) &&
      typeof this.CacheEngines?.setter === "function"
    )
      this.CacheEngines.setter(
        `TRN_${this.Options.defaultLanguage}_${this.Options.currentLangauge}_` +
          key,
        value
      );
  }

  public async getCache(key: string) {
    if (
      [undefined, true].includes(this.Options.cache) &&
      typeof this.CacheEngines?.getter === "function"
    )
      return this.CacheEngines.getter(
        `TRN_${this.Options.defaultLanguage}_${this.Options.currentLangauge}_` +
          key
      );
    else return null;
  }

  public async translate<T extends string | string[]>(
    targets?: T
  ): Promise<T extends string ? string : Record<string, string>> {
    // Resolve Targets
    if (targets === undefined) targets = this.TranslationTargets as T;

    // Collect Translations from Cache
    let Translated: Record<string, string> = {
      ...(
        await Promise.all(
          (targets instanceof Array ? targets : [targets]).map(
            async (target) => {
              const value = await this.getCache(target);
              if (value || target === "")
                return {
                  key: target,
                  value: target === "" ? target : value,
                };
              else return undefined;
            }
          )
        )
      )
        .filter((v) => v !== undefined)
        .reduce(
          (object, item) => ({
            ...object,
            [item!.key]: item!.value,
          }),
          {}
        ),
    };

    // Get Translated Targets
    const TranslatedTargets = Object.keys(Translated);

    // Filter Remaining Targets
    const RemainingTranslations = (targets instanceof Array
      ? targets
      : [targets]
    ).filter((target) => !TranslatedTargets.includes(target));

    // Check Remaining Length
    if (RemainingTranslations.length) {
      try {
        // Fetch New Translations
        const Results =
          this.Options.defaultLanguage !== this.Options.currentLangauge
            ? await this.Options.engine(
                {
                  default: this.Options.defaultLanguage,
                  current: this.Options.currentLangauge,
                },
                RemainingTranslations
              )
            : RemainingTranslations.reduce<TranslationResponseShape>(
                (object, target) => ({ ...object, [target]: target }),
                {}
              );

        if (typeof Results === "object") {
          // Store Translation to Cache
          if (this.Options.defaultLanguage !== this.Options.currentLangauge)
            await Promise.all(
              Object.keys(Results).map((target) =>
                this.setCache(target, Results[target])
              )
            );

          // Combine Results
          Translated = {
            ...Translated,
            ...Results,
          };
        }
      } catch (e) {
        // Combine Results
        Translated = {
          ...Translated,
          ...RemainingTranslations.reduce<TranslationResponseShape>(
            (object, target) => ({ ...object, [target]: target }),
            {}
          ),
        };
      }
    }

    // Store Results
    this.TranslationResults = ((targets instanceof Array
      ? targets
      : [targets]) as string[]).reduce(
      (object, target) => ({
        ...object,
        [target]: Translated[target],
      }),
      {}
    );

    // Return Results
    return (typeof targets === "string"
      ? Object.values(this.TranslationResults)[0]
      : this.TranslationResults) as any;
  }
}
