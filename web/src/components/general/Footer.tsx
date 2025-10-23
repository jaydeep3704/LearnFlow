import { GraduationCap } from "lucide-react";

export function Footer(){
    return(
        <section className="px-[4%] py-5 flex justify-between items-center border-t border-muted-foreground/15 shadow-xs">
             <h1 className="flex gap-2 items-center font-bold">
                    <GraduationCap className="text-primary size-6" />
                    <span className=" lg:text-2xl">LearnFlow</span>
            </h1>
            <p className="text-muted-foreground lg:text-md text-sm">© 2025 LearnFlow. AI-powered learning platform.</p>
        </section>
    )
}