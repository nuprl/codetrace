class __typ0 extends Error {};

declare module 'assert' {
    
    function assert(value, message?: string | __typ0): void;
    namespace assert {
        class AssertionError implements __typ0 {
            name: string;
            message: string;
            actual: any;
            expected: any;
            operator: string;
            generatedMessage: boolean;
            code: 'ERR_ASSERTION';

            constructor(options?: {
                
                message?: string;
                
                actual?: any;
                
                expected?: any;
                
                operator?: string;
                
                
                stackStartFn?: Function;
            });
        }

        class __typ1 {
            calls(exact?: number): () => void;
            calls<Func extends (...args: any[]) => any>(fn?: Func, exact?: number): Func;
            report(): CallTrackerReportInformation[];
            verify(): void;
        }
        interface CallTrackerReportInformation {
            message: string;
            
            actual: number;
            
            expected: number;
            
            operator: string;
            
            stack: object;
        }

        type AssertPredicate = RegExp | (new () => object) | ((thrown) => boolean) | object | __typ0;

        function fail(message?: string | __typ0): never;
        
        function fail(
            actual,
            expected,
            message?: string | __typ0,
            operator?: string,
            
            stackStartFn?: Function,
        ): never;
        function ok(value, message?: string | __typ0): void;
        
        function equal(actual, expected, message?: string | __typ0): void;
        
        function notEqual(actual, expected, message?: string | __typ0): void;
        
        function deepEqual(actual, expected, message?: string | __typ0): void;
        
        function notDeepEqual(actual: <FILL>, expected, message?: string | __typ0): void;
        function strictEqual(actual, expected, message?: string | __typ0): void;
        function notStrictEqual(actual, expected, message?: string | __typ0): void;
        function deepStrictEqual(actual, expected, message?: string | __typ0): void;
        function notDeepStrictEqual(actual, expected, message?: string | __typ0): void;

        function __tmp1(__tmp0: () => any, message?: string | __typ0): void;
        function __tmp1(__tmp0: () => any, error: AssertPredicate, message?: string | __typ0): void;
        function doesNotThrow(__tmp0: () => any, message?: string | __typ0): void;
        function doesNotThrow(__tmp0: () => any, error: AssertPredicate, message?: string | __typ0): void;

        function ifError(value): void;

        function rejects(__tmp0: (() => Promise<any>) | Promise<any>, message?: string | __typ0): Promise<void>;
        function rejects(
            __tmp0: (() => Promise<any>) | Promise<any>,
            error: AssertPredicate,
            message?: string | __typ0,
        ): Promise<any>;
        function doesNotReject(__tmp0: (() => Promise<any>) | Promise<any>, message?: string | __typ0): Promise<void>;
        function doesNotReject(
            __tmp0: (() => Promise<any>) | Promise<any>,
            error: AssertPredicate,
            message?: string | __typ0,
        ): Promise<any>;

        function match(value: string, regExp: RegExp, message?: string | __typ0): void;
        function doesNotMatch(value: string, regExp: RegExp, message?: string | __typ0): void;

        const strict: typeof assert;
        function id(val: __typ1): __typ1;
    }

    export = assert;
}
function __tmp1(val: __typ0) {
    throw new __typ0('Something went wrong');
}
