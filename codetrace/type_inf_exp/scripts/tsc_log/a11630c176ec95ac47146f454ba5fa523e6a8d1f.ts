declare interface ObjectCtor extends ObjectConstructor {
	assign<T>(target: {}, ...source: Array<Partial<T>>): T;

	entries<T, Obj>(o: Obj): Array<[Extract<keyof Obj, string>, T]>;
}


declare let Object: ObjectCtor;


export type ObjMap<Obj, T> = { [name in Extract<keyof Obj, string>]: T };

export type ObjMapPart<Obj, T> = Partial<ObjMap<Obj, T>>;


export type KeyMapper<T> = (value: T) => string;


export function mapObject<T, O, Obj = Record<string, T>>(obj: Obj, fn: (value: T, key: Extract<keyof Obj, string>) => O) {
	
	const mapped = Object.entries<T, Obj>(obj).map(([key, value]: [Extract<keyof Obj, string>, T]) => ({ [key]: fn(value, key) } as ObjMapPart<Obj, O>));
	return Object.assign<ObjMap<Obj, O>>({}, ...mapped);
}


export function objectFromArray<T, O, Obj>(arr: T[], fn: (value: T) => ObjMapPart<Obj, O>) {
	return Object.assign<ObjMap<Obj, O>>({}, ...arr.map(fn));
}


export function indexBy<T>(arr: T[], key: Extract<keyof T, string>): Record<string, T>;

export function indexBy<T>(arr: T[], keyFn: KeyMapper<T>): Record<string, T>;

export function indexBy<T>(arr: T[], keyFn: Extract<keyof T, string> | KeyMapper<T>): Record<string, T> {
	if (typeof keyFn !== 'function') {
		const key = keyFn;
		keyFn = ((value: T) => (value[key] as unknown as object).toString()) as KeyMapper<T>;
	}
	return objectFromArray<T, T, Record<string, T>>(arr, val => ({ [(keyFn as KeyMapper<T>)(val)]: val }));
}


export function forEachObjectEntry<T, Obj>(obj: Obj, fn: (value: T, key: Extract<keyof Obj, string>) => void) {
	Object.entries(obj).forEach(([key, value]: [Extract<keyof Obj, string>, T]) => fn(value, key));
}


export function entriesToObject<T>(obj: Array<[string, T]>): Record<string, T> {
	return objectFromArray<[string, T], T, Record<string, T>>(obj, ([key, val]) => ({ [key]: val }));
}