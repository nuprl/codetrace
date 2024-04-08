declare const _default: {
    string: {
        number: {
            test: __typ0;
            convert: __typ0;
        };
        boolean: {
            test: (value: any) => boolean;
            convert: (value: any) => boolean;
        };
        richDate: {
            test: (val: any) => boolean;
            convert: (value: any) => {
                _type: string;
                local: string;
                utc: string;
                timezone: string;
                offset: number;
            };
        };
    };
    date: {
        richDate: {
            test: (val: any) => boolean;
            convert: (value: any) => any;
        };
    };
    richDate: {
        datetime: {
            test: (val: any) => any;
            convert: (value: any) => any;
        };
    };
    number: {
        string: {
            test: () => boolean;
            convert: StringConstructor;
        };
        boolean: {
            test: () => boolean;
            convert: __typ0;
        };
    };
    boolean: {
        string: {
            test: () => boolean;
            convert: (value: any) => "Yes" | "No";
        };
        number: {
            test: () => boolean;
            convert: __typ0;
        };
    };
};
export default _default;