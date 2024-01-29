// This file was autogenerated. Please do not change.
// All changes will be overwrited on commit.
export interface IOrganization {
    description?: string;
    id?: <FILL>;
    image?: string;
    name?: string;
    website?: string;
}

export default class Organization {
    readonly _description: string | undefined;

    get description(): string | undefined {
        return this._description;
    }

    readonly _id: string | undefined;

    get id(): string | undefined {
        return this._id;
    }

    readonly _image: string | undefined;

    get image(): string | undefined {
        return this._image;
    }

    readonly _name: string | undefined;

    get name(): string | undefined {
        return this._name;
    }

    readonly _website: string | undefined;

    get website(): string | undefined {
        return this._website;
    }

    constructor(props: IOrganization) {
        if (typeof props.description === 'string') {
            this._description = props.description.trim();
        }
        if (typeof props.id === 'string') {
            this._id = props.id.trim();
        }
        if (typeof props.image === 'string') {
            this._image = props.image.trim();
        }
        if (typeof props.name === 'string') {
            this._name = props.name.trim();
        }
        if (typeof props.website === 'string') {
            this._website = props.website.trim();
        }
    }

    serialize(): IOrganization {
        const data: IOrganization = {};
        if (typeof this._description !== 'undefined') {
            data.description = this._description;
        }
        if (typeof this._id !== 'undefined') {
            data.id = this._id;
        }
        if (typeof this._image !== 'undefined') {
            data.image = this._image;
        }
        if (typeof this._name !== 'undefined') {
            data.name = this._name;
        }
        if (typeof this._website !== 'undefined') {
            data.website = this._website;
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
            name: !this._name
                ? true
                : typeof this._name === 'string' && !this._name
                ? true
                : this._name,
            description: !this._description
                ? true
                : typeof this._description === 'string' && !this._description
                ? true
                : this._description,
            image: !this._image
                ? true
                : typeof this._image === 'string' && !this._image
                ? true
                : this._image,
            website: !this._website
                ? true
                : typeof this._website === 'string' && !this._website
                ? true
                : this._website,
        };
        const isError: string[] = [];
        Object.keys(validate).forEach((key) => {
            if (!(validate as any)[key]) {
                isError.push(key);
            }
        });
        return isError;
    }

    update(props: Partial<IOrganization>): Organization {
        return new Organization({...this.serialize(), ...props});
    }
}
