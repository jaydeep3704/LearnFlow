import { Button, buttonVariants } from "@/components/ui/button";
import { GraduationCap, LogOutIcon } from "lucide-react";
import Link from "next/link";
import { ModeToggle } from "./ModeToggle";
import { auth, signOut } from "@/utils/auth";
import { Avatar, AvatarFallback, AvatarImage } from "../ui/avatar";
import {
    DropdownMenu,
    DropdownMenuContent,
    DropdownMenuGroup,
    DropdownMenuItem,
    DropdownMenuLabel,
    DropdownMenuPortal,
    DropdownMenuSeparator,
    DropdownMenuShortcut,
    DropdownMenuSub,
    DropdownMenuSubContent,
    DropdownMenuSubTrigger,
    DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu"


interface iUserProps {
    email: string,
    id: string,
    image: string,
    name: string
}

export async function Navbar() {
    const session = await auth()
    const user = await session?.user
    return (
        <div
            className="px-[4%] py-5 shadow-sm border-b border-muted-foreground/15 sticky top-0 left-0 right-0 bg-background/50
        backdrop-blur-md z-100
        ">
            <div className="flex justify-between">
                <h1 className="flex gap-2 items-center font-bold">
                    <GraduationCap className="text-primary size-6" />
                    <span className=" text-2xl">LearnFlow</span>
                </h1>
                <div className="flex gap-4 items-center">
                    <ModeToggle />

                    { (session && user ) ? (<div className="flex gap-4">
                        <Link
                            href={"/courses"}
                            className={buttonVariants({ variant: "default", className: "font-semibold" })}>
                            My Courses
                        </Link>
                        <UserAvatar user={user} />
                    </div>):
                    <div>
                         <Link
                            href={"/login"}
                            className={buttonVariants({ variant: "default", className: "font-semibold px-8" })}>
                            Login
                        </Link>
                    </div>
                    }
                </div>
            </div>
        </div>
    )
}


function UserAvatar({ user }: { user: iUserProps }) {
    return (<>
        <DropdownMenu>
            <DropdownMenuTrigger asChild>
                <Avatar >
                    <AvatarFallback>{user.name?.split('').at(0)?.at(0) as string}</AvatarFallback>
                    <AvatarImage src={user.image as string} alt={user.name as string} />
                </Avatar>
            </DropdownMenuTrigger>
            <DropdownMenuContent className="z-[110]" align="end">
                <DropdownMenuLabel className="text-sm">{user.email}</DropdownMenuLabel>
                <DropdownMenuSeparator/>
                <DropdownMenuGroup>
                   <form action={async()=>{
                      "use server"
                       await signOut()
                   }}>
                        <Button className="text-sm flex items-center justify-between w-full" variant={"outline"}>
                        Signout <LogOutIcon/>
                        </Button>
                    </form> 
                    
                </DropdownMenuGroup>
            </DropdownMenuContent>
        </DropdownMenu>


    </>)
}