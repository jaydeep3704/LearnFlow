import { buttonVariants } from "@/components/ui/button";
import { GraduationCap } from "lucide-react";
import Link from "next/link";
import { ModeToggle } from "./ModeToggle";

export function Navbar() {
    return (
        <div 
        className="px-[4%] py-5 shadow-sm border-b border-muted-foreground/15 sticky top-0 left-0 right-0 bg-background/50
        backdrop-blur-md 
        ">
            <div className="flex justify-between">
                <h1 className="flex gap-2 items-center font-bold">
                    <GraduationCap className="text-primary size-6" />
                    <span className=" text-2xl">LearnFlow</span>
                </h1>
                <div className="flex gap-4">
                    <ModeToggle/>
                    <Link
                        href={"/courses"}
                        className={buttonVariants({ variant: "default", className: "font-semibold" })}>
                        My Courses
                    </Link>
                </div>
            </div>
        </div>
    )
}