// This file was autogenerated. Please do not change.
// All changes will be overwrited on commit.
export interface IUser {
    admin?: boolean;
    date_password?: string;
    email?: <FILL>;
    enabled?: boolean;
    id?: string;
    username?: string;
}

export default class User {
    readonly _admin: boolean | undefined;

    get admin(): boolean | undefined {
        return this._admin;
    }

    readonly _date_password: string | undefined;

    get datePassword(): string | undefined {
        return this._date_password;
    }

    readonly _email: string | undefined;

    get email(): string | undefined {
        return this._email;
    }

    readonly _enabled: boolean | undefined;

    get enabled(): boolean | undefined {
        return this._enabled;
    }

    readonly _id: string | undefined;

    get id(): string | undefined {
        return this._id;
    }

    readonly _username: string | undefined;

    get username(): string | undefined {
        return this._username;
    }

    constructor(props: IUser) {
        if (typeof props.admin === 'boolean') {
            this._admin = props.admin;
        }
        if (typeof props.date_password === 'string') {
            this._date_password = props.date_password.trim();
        }
        if (typeof props.email === 'string') {
            this._email = props.email.trim();
        }
        if (typeof props.enabled === 'boolean') {
            this._enabled = props.enabled;
        }
        if (typeof props.id === 'string') {
            this._id = props.id.trim();
        }
        if (typeof props.username === 'string') {
            this._username = props.username.trim();
        }
    }

    serialize(): IUser {
        const data: IUser = {};
        if (typeof this._admin !== 'undefined') {
            data.admin = this._admin;
        }
        if (typeof this._date_password !== 'undefined') {
            data.date_password = this._date_password;
        }
        if (typeof this._email !== 'undefined') {
            data.email = this._email;
        }
        if (typeof this._enabled !== 'undefined') {
            data.enabled = this._enabled;
        }
        if (typeof this._id !== 'undefined') {
            data.id = this._id;
        }
        if (typeof this._username !== 'undefined') {
            data.username = this._username;
        }
        return data;
    }

    validate(): string[] {
        const validate = {
            id: !this._id
                ? true
                : typeof this._id === 'string' && !this._id
                ? true
                : this._id,
            username: !this._username
                ? true
                : typeof this._username === 'string' && !this._username
                ? true
                : this._username,
            email: !this._email
                ? true
                : typeof this._email === 'string' && !this._email
                ? true
                : this._email,
            date_password: !this._date_password
                ? true
                : typeof this._date_password === 'string' &&
                  !this._date_password
                ? true
                : this._date_password,
            enabled: !this._enabled ? true : typeof this._enabled === 'boolean',
            admin: !this._admin ? true : typeof this._admin === 'boolean',
        };
        const isError: string[] = [];
        Object.keys(validate).forEach((key) => {
            if (!(validate as any)[key]) {
                isError.push(key);
            }
        });
        return isError;
    }

    update(props: Partial<IUser>): User {
        return new User({...this.serialize(), ...props});
    }
}
