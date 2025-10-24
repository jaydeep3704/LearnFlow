import { CreateCourseButton } from "@/components/general/Course/CreateCourseButton"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Dialog } from "@/components/ui/dialog"
import { auth } from "@/utils/auth"
import { PlusIcon } from "lucide-react"
import { redirect } from "next/navigation"


export default async function Courses() {
    const session = await auth()
    if (!session) redirect("/login")

    return (
        <section className="min-h-screen w-full px-[4%] py-4">
            <Card
                className={`py-20 max-w-7xl h-[400px] bg-[url(/illustration.jpg)]  bg-cover bg-center mx-auto relative rounded-3xl overflow-hidden`}
            >
                  <div className="absolute inset-0 bg-black/60"></div> {/* 40% black overlay */}
                {/* Add your content here */}
                <div className="z-50">
                <CardHeader className="">
                    <CardTitle className="text-center font-bold lg:text-5xl text-2xl drop-shadow-md text-white">Welcome Back !</CardTitle>
                    <CardDescription className="text-center lg:text-2xl text-lg font-bold text-gray-400 drop-shadow-md ">
                        Continue your learning journey or create a new course from any YouTube video
                    </CardDescription>
                </CardHeader>
                <CardContent className="flex justify-center items-center py-5 ">
                    <CreateCourseButton/>
                </CardContent>
                </div>
            </Card>
        </section>
    )
}





