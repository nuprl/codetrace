declare module 'assert' {
    
    function assert(value: any, message?: string | Error): void;
    namespace assert {
        class AssertionError implements Error {
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

        class CallTracker {
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

        type AssertPredicate = RegExp | (new () => object) | ((thrown: any) => boolean) | object | Error;

        function fail(message?: string | Error): never;
        
        function fail(
            actual: any,
            expected: any,
            message?: string | Error,
            operator?: string,
            
            stackStartFn?: Function,
        ): never;
        function ok(value: any, message?: string | Error): void;
        
        function equal(actual: any, expected: any, message?: string | Error): void;
        
        function notEqual(actual: any, expected: any, message?: string | Error): void;
        
        function deepEqual(actual: any, expected: any, message?: string | Error): void;
        
        function notDeepEqual(actual: <FILL>, expected: any, message?: string | Error): void;
        function strictEqual(actual: any, expected: any, message?: string | Error): void;
        function notStrictEqual(actual: any, expected: any, message?: string | Error): void;
        function deepStrictEqual(actual: any, expected: any, message?: string | Error): void;
        function notDeepStrictEqual(actual: any, expected: any, message?: string | Error): void;

        function throws(block: () => any, message?: string | Error): void;
        function throws(block: () => any, error: AssertPredicate, message?: string | Error): void;
        function doesNotThrow(block: () => any, message?: string | Error): void;
        function doesNotThrow(block: () => any, error: AssertPredicate, message?: string | Error): void;

        function ifError(value: any): void;

        function rejects(block: (() => Promise<any>) | Promise<any>, message?: string | Error): Promise<void>;
        function rejects(
            block: (() => Promise<any>) | Promise<any>,
            error: AssertPredicate,
            message?: string | Error,
        ): Promise<any>;
        function doesNotReject(block: (() => Promise<any>) | Promise<any>, message?: string | Error): Promise<void>;
        function doesNotReject(
            block: (() => Promise<any>) | Promise<any>,
            error: AssertPredicate,
            message?: string | Error,
        ): Promise<any>;

        function match(value: string, regExp: RegExp, message?: string | Error): void;
        function doesNotMatch(value: string, regExp: RegExp, message?: string | Error): void;

        const strict: typeof assert;
        function id(val: CallTracker): CallTracker;
    }

    export = assert;
}
function throws(val: Error) {
    throw new Error('Something went wrong');
}
