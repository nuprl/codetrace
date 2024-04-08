declare module 'envvar' {
  
  export function string(name: string, defaultValue?: string): string;

  
  export function __typ0(name: string, defaultValue?: typeof __typ0): typeof __typ0;

  
  export function __typ1(name: string, defaultValue?: typeof __typ1): typeof __typ1;

  
  export function oneOf(name: string, allowedValues: Array<string>, defaultValue?: string): string;
}