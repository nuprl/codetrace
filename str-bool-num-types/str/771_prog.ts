export interface MemberJSON {
    id: number
    member_id: number
    name: string
    email: string
    mobile_number: string
    gender: string
    birth_date: Date
    address: string
    city: string
    zip_code: number
    image_profile: string
    image_card: string
    marital_status: string
    membershipType: string
    points: number
    register_date: Date
    expired_date: Date
    remark: string
    is_active: boolean
}

export class Member {
    protected id: number
    protected member_id: number
    protected name: string
    protected email: string
    protected mobile_number: string
    protected gender: string
    protected birth_date: Date
    protected address: string
    protected city: string
    protected zip_code: number
    protected image_profile: string
    protected image_card: string
    protected marital_status: <FILL>
    protected membershipType: string
    protected points: number
    protected register_date: Date
    protected expired_date: Date
    protected remark: string
    protected is_active: boolean

    fromJSON(json: MemberJSON) {
        this.id = json.id,
        this.member_id = json.member_id,
        this.name = json.name,
        this.email = json.email,
        this.mobile_number = json.mobile_number,
        this.gender = json.gender,
        this.birth_date = json.birth_date,
        this.address = json.address,
        this.city = json.city,
        this.zip_code = json.zip_code,
        this.image_profile = json.image_profile,
        this.image_card = json.image_card,
        this.marital_status = json.marital_status,
        this.membershipType = json.membershipType,
        this.points = json.points,
        this.register_date = json.register_date,
        this.expired_date = json.expired_date,
        this.remark = json.remark,
        this.is_active = json.is_active
    }

    toJSON(): MemberJSON {
        return {
            id: this.id,
            member_id: this.member_id,
            name: this.name,
            email: this.email,
            mobile_number: this.mobile_number,
            gender: this.gender,
            birth_date: this.birth_date,
            address: this.address,
            city: this.city,
            zip_code: this.zip_code,
            image_profile: this.image_profile,
            image_card: this.image_card,
            marital_status: this.marital_status,
            membershipType: this.membershipType,
            points: this.points,
            register_date: this.register_date,
            expired_date: this.expired_date,
            remark: this.remark,
            is_active: this.is_active
        }
    }
}