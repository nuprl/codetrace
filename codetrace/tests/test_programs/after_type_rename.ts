type __typ1 = string;

export class __typ0 {
    private _id: __typ1;
    private _fullName: __typ1;
    private _password: __typ1;
    private _state: __typ1;
    private _roles: __typ1[];

    constructor(fullName?: __typ1, roles?: __typ1[]) {
        this._fullName = fullName;
        this._roles = roles;
    }

    public get id(): __typ1 {
        return this._id;
    }

    public get fullName(): __typ1 {
        return this._fullName;
    }

    public get password(): __typ1 {
        return this._password;
    }

    public get state(): __typ1 {
        return this._state;
    }

    public get roles(): __typ1[] {
        return this._roles;
    }

    public set id(value:__typ1) {
        this._id = value;
    }

    public set fullName(value: <FILL>) {
        this._fullName = value;
    }

    public set password(value:__typ1) {
        this.password = value;
    }

    public set state(value:__typ1) {
        this._state = value;
    }

    public set roles(value:__typ1[]) {
        this._roles = value;
    }

    public static fromJSON(rawUser : any) : __typ0 {
        const tmpUser = new __typ0(rawUser['fullname'], rawUser['roles']);
        tmpUser.id = rawUser['_id'];
        tmpUser.state = rawUser['state'];
        tmpUser.password = rawUser['password'];
        return tmpUser;
    }

    public static fromArrayJSON(rawUsers : any[]) : __typ0[] {
        return rawUsers.map(__typ0.fromJSON);
    }

    public getCleanDataForSending() {
        return {
            "_id" : this._id,
            "fullname": this._fullName,
            "password": this._password,
            "state": this._state,
            "roles" : this._roles
        };
    }
}
